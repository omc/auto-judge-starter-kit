#!/usr/bin/env python3
"""OpenRouter Batch API adapter.

minima's BatchCollector speaks the OpenAI file-upload batch protocol (POST /files,
POST /batches with input_file_id, GET /files/{id}/content). OpenRouter's batch API
diverges: no file upload, an inline `requests` array, and results returned inline
from GET /api/beta/batches/:id. See https://openrouter.ai/docs/batch-quickstart.

This adapter submits comparisons via OpenRouter's protocol and populates the SAME
minima PromptCache (keyed identically), so the rest of the pairwise judge -- which
reads results back out of the cache by key -- is unchanged. Batch pricing is ~50%
of realtime; the 24h completion window applies.

Resumable: a JSON state file per prefix records each chunk's batch_id and its
custom_id->cache_key map. Re-running fetches completed chunks and keeps polling
active ones; already-cached comparisons are never resubmitted.
"""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from minima_llm import MinimaLlmConfig, MinimaLlmRequest

Json = Dict[str, Any]
TERMINAL = {"completed", "failed", "expired", "cancelled"}


def derive_batches_url(base_url: str) -> str:
    """OpenRouter batches live under /api/beta/batches, not the /api/v1 chat base."""
    b = base_url.rstrip("/")
    if "/api/v1" in b:
        b = b.split("/api/v1")[0] + "/api"
    elif b.endswith("/v1"):
        b = b[: -len("/v1")]
    return b + "/beta/batches"


class OpenRouterBatch:
    def __init__(
        self,
        cfg: MinimaLlmConfig,
        cache: Any,                       # minima PromptCache (get/put)
        state_dir: Path,
        prefix: str,
        *,
        poll_interval_s: float = 30.0,
        max_poll_hours: float = 24.0,
        max_requests_per_batch: int = 5000,   # OpenRouter caps a batch at 5,000 requests
        max_inflight_requests: int = 20000,    # OpenRouter caps in-flight requests/entity
        max_status_retries: int = 30,          # tolerate transient GET 404/5xx per chunk
        batches_url: Optional[str] = None,
        transport: Optional[Callable[[str, str, Optional[bytes], Dict[str, str]], Json]] = None,
    ):
        self.cfg = cfg
        self.cache = cache
        self.prefix = prefix
        self.poll_interval_s = poll_interval_s
        self.max_poll_hours = max_poll_hours
        self.max_requests_per_batch = max_requests_per_batch
        self.max_inflight_requests = max_inflight_requests
        self.max_status_retries = max_status_retries
        self.batches_url = batches_url or derive_batches_url(cfg.base_url)
        self._transport = transport or self._http
        state_dir = Path(state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        import re as _re
        safe = _re.sub(r"[^A-Za-z0-9_.-]", "_", prefix)   # never let '/' into the filename
        self.state_file = state_dir / f"openrouter_batch_{safe}.json"

    # ---- request body ----

    def _body(self, req: MinimaLlmRequest) -> Json:
        body: Json = {"messages": req.messages}
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if req.max_tokens is not None:
            body["max_tokens"] = req.max_tokens
        if req.extra:
            body.update(req.extra)
        return body

    def _create_payload(self, chunk: List[Tuple[MinimaLlmRequest, str, str]]) -> bytes:
        """chunk = [(req, cache_key, custom_id)]. Field order MUST be endpoint, model,
        requests: OpenRouter stream-parses and 400s if `requests` comes first."""
        payload = {
            "endpoint": "/v1/chat/completions",
            "model": self.cfg.model,
            "requests": [{"custom_id": cid, "body": self._body(req)} for req, _k, cid in chunk],
        }
        return json.dumps(payload).encode("utf-8")

    # ---- HTTP ----

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            h["Authorization"] = f"Bearer {self.cfg.api_key}"
        return h

    def _http(self, method: str, url: str, body: Optional[bytes], headers: Dict[str, str]) -> Json:
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            txt = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code} from {method} {url}: {txt[:500]}")

    def _create(self, chunk) -> str:
        resp = self._transport("POST", self.batches_url, self._create_payload(chunk), self._headers())
        bid = resp.get("id") or resp.get("batch_id")
        if not bid:
            raise RuntimeError(f"OpenRouter batch create returned no id: {str(resp)[:300]}")
        return bid

    def _status(self, batch_id: str) -> Json:
        return self._transport("GET", f"{self.batches_url}/{batch_id}", None, self._headers())

    def _inflight_requests(self) -> int:
        """Sum request_counts.total over all this entity's non-terminal batches.
        OpenRouter enforces a 20k in-flight-requests-per-entity cap, so we must not
        create a batch that would push over it."""
        total = 0
        resp = self._transport("GET", f"{self.batches_url}?limit=100", None, self._headers())
        for b in (resp.get("data") or []):
            if b.get("status") not in TERMINAL:
                total += int((b.get("request_counts") or {}).get("total", 0) or 0)
        return total

    # ---- state ----

    def _load_state(self) -> Optional[dict]:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return None

    def _save_state(self, state: dict) -> None:
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(self.state_file)

    # ---- in-flight duplicate guard (shared across prefixes) ----
    # A cache-dir-wide index mapping a chunk's content signature (hash of its sorted
    # cache keys) to the batch already submitted for it. Prevents resubmitting an
    # identical chunk even if the state prefix changes -- the failure mode that minted
    # duplicate 5,000-request batches during debugging.

    @property
    def _index_file(self) -> Path:
        return self.state_file.parent / "openrouter_submitted.json"

    def _load_index(self) -> Dict[str, dict]:
        if self._index_file.exists():
            try:
                return json.loads(self._index_file.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_index(self, idx: Dict[str, dict]) -> None:
        tmp = self._index_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(idx))
        tmp.replace(self._index_file)

    @staticmethod
    def _chunk_sig(chunk) -> str:
        import hashlib as _h
        return _h.sha256("\n".join(sorted(key for _r, key, _c in chunk)).encode()).hexdigest()

    # ---- results ----

    def _populate_from_status(self, info: Json, id2key: Dict[str, str]) -> Tuple[int, int]:
        """Write completed results inline from a batch status response into cache."""
        results = info.get("results") or info.get("output") or []
        ok = fail = 0
        for item in results:
            cid = item.get("custom_id")
            key = id2key.get(cid)
            if not key:
                continue
            if item.get("error"):
                fail += 1
                continue
            body = (item.get("response") or {}).get("body") or {}
            choices = body.get("choices") or []
            if not choices:
                fail += 1
                continue
            text = (choices[0].get("message") or {}).get("content", "")
            self.cache.put(key, text, body)
            ok += 1
        return ok, fail

    # ---- main ----

    # ---- decoupled phases: submit (create, non-blocking) then harvest (poll) ----

    def submit(self, items: List[Tuple[MinimaLlmRequest, str]]) -> dict:
        """Create batches for every uncached, deduped item and return IMMEDIATELY
        (no polling). Idempotent: chunks already recorded in state are not recreated,
        so re-running never double-submits."""
        seen: set = set()
        pending: List[Tuple[MinimaLlmRequest, str, str]] = []
        cached = 0
        for req, key in items:
            if self.cache.get(key) is not None:
                cached += 1
                continue
            if key in seen:
                continue
            seen.add(key)
            pending.append((req, key, f"c{len(pending)}"))

        state = self._load_state() or {"prefix": self.prefix, "chunks": []}
        existing = {c["index"]: c for c in state["chunks"]}

        adopted = created = 0
        submitted_all = True
        if pending:
            n = self.max_requests_per_batch
            chunks = [pending[i:i + n] for i in range(0, len(pending), n)]
            index = self._load_index()
            budget = None                         # lazily query in-flight only if creating
            for ci, chunk in enumerate(chunks):
                if ci in existing and existing[ci].get("batch_id"):
                    continue                      # already in this state; never recreate
                sig = self._chunk_sig(chunk)
                prior = index.get(sig)
                if prior:
                    # Identical chunk already submitted (possibly under another prefix).
                    # Adopt its batch rather than pay to submit a duplicate. Already
                    # counted in the in-flight total, so it costs no budget.
                    existing[ci] = {"index": ci, "batch_id": prior["batch_id"],
                                    "status": "validating", "id2key": prior["id2key"]}
                    adopted += 1
                    print(f"[openrouter] chunk {ci+1}/{len(chunks)}: adopted in-flight "
                          f"batch {prior['batch_id']} (dup guard; identical content)")
                    state["chunks"] = [existing[k] for k in sorted(existing)]
                    self._save_state(state)
                    continue

                # Need to create -> respect the 20k in-flight-requests cap.
                if budget is None:
                    budget = self.max_inflight_requests - self._inflight_requests()
                if budget < len(chunk):
                    submitted_all = False
                    print(f"[openrouter] in-flight budget {budget} < chunk {len(chunk)}; "
                          f"deferring {len(chunks) - ci} chunk(s) -- re-run after batches drain")
                    break
                try:
                    bid = self._create(chunk)
                except RuntimeError as e:
                    if "429" in str(e):            # raced another submitter to the cap
                        submitted_all = False
                        print("[openrouter] 429 in-flight cap; deferring remaining chunks")
                        break
                    raise
                budget -= len(chunk); created += 1
                id2key = {cid: key for _r, key, cid in chunk}
                existing[ci] = {"index": ci, "batch_id": bid, "status": "validating",
                                "id2key": id2key}
                index[sig] = {"batch_id": bid, "id2key": id2key}
                self._save_index(index)
                print(f"[openrouter] created batch chunk {ci+1}/{len(chunks)}: "
                      f"{bid} ({len(chunk)} reqs)")
                state["chunks"] = [existing[k] for k in sorted(existing)]
                self._save_state(state)

        return {"cached": cached, "submitted": len(pending), "chunks": len(existing),
                "adopted": adopted, "created": created, "submitted_all": submitted_all}

    def harvest(self, wait: bool = False, poll_timeout_s: float = 0.0) -> Tuple[bool, dict]:
        """Check this prefix's in-flight batches, writing completed results into the
        cache. wait=False does a SINGLE status pass then returns (decoupled mode --
        re-run later to pick up more). wait=True loops until all batches are terminal
        or poll_timeout_s (0 => max_poll_hours) elapses.

        Returns (all_terminal, summary). all_terminal=True => every chunk finished
        (completed/failed/expired/cancelled), i.e. safe to score."""
        state = self._load_state()
        if not state or not state.get("chunks"):
            return True, {"chunks": 0, "terminal": 0, "completed": 0, "failed": 0, "pending": 0}
        existing = {c["index"]: c for c in state["chunks"]}
        cap_s = poll_timeout_s or self.max_poll_hours * 3600
        start = time.time()
        status_fails: Dict[int, int] = {}
        ok = fail = 0
        while True:
            active = [ci for ci in existing if existing[ci].get("status") not in TERMINAL]
            for ci in list(active):
                cs = existing[ci]
                try:
                    info = self._status(cs["batch_id"])
                except Exception as e:
                    # Freshly-created batches 404 on GET /:id briefly (eventual
                    # consistency); transient 5xx too. Tolerate, don't die.
                    status_fails[ci] = status_fails.get(ci, 0) + 1
                    if status_fails[ci] > self.max_status_retries:
                        raise RuntimeError(f"chunk {ci} ({cs['batch_id']}) status failed "
                                           f"{status_fails[ci]}x: {e}")
                    print(f"[openrouter] chunk {ci} status transient error "
                          f"({status_fails[ci]}/{self.max_status_retries}): {e}")
                    continue
                status_fails[ci] = 0
                status = info.get("status", "unknown")
                cs["status"] = status
                if status == "completed":
                    o, f = self._populate_from_status(info, cs["id2key"])
                    ok += o; fail += f
                    print(f"[openrouter] chunk {ci} completed: {o} ok, {f} failed")
                elif status in TERMINAL:
                    print(f"[openrouter] chunk {ci} {status} (no results)")
            state["chunks"] = [existing[k] for k in sorted(existing)]
            self._save_state(state)

            active = [ci for ci in existing if existing[ci].get("status") not in TERMINAL]
            if not active or not wait or (time.time() - start) > cap_s:
                break
            counts = ", ".join(f"c{ci}:{existing[ci]['status']}" for ci in sorted(active))
            print(f"[openrouter] {len(active)} active ({(time.time()-start)/60:.0f}m) - {counts}")
            time.sleep(self.poll_interval_s)

        active = [ci for ci in existing if existing[ci].get("status") not in TERMINAL]
        return (not active), {"chunks": len(existing), "terminal": len(existing) - len(active),
                              "completed": ok, "failed": fail, "pending": len(active)}

    def submit_and_cache(self, items: List[Tuple[MinimaLlmRequest, str]]) -> dict:
        """Convenience: submit then block-poll until done (the coupled path)."""
        s = self.submit(items)
        ready, h = self.harvest(wait=True)
        return {**s, **h, "ready": ready}
