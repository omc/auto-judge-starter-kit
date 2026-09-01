# The base image is build from .devcontainer/Dockerfile
FROM ghcr.io/trec-auto-judge/trec-auto-judge-base:dev-0.0.1

ADD judges /auto-judge/judges
ADD pyproject.toml /auto-judge/

WORKDIR /auto-judge

# Install into the base image's /venv (its PATH runs /venv/bin) — a --system
# install would be invisible at runtime, breaking any dependency added here.
RUN . /venv/bin/activate && uv pip install -e .[all]

# spaCy model for concept-F1 (bonsai_judge). The model is not a pip dependency;
# `spacy download` fetches the wheel matching the installed spaCy version.
# For the transformer variant use en_core_web_trf instead (also needs
# spacy-transformers + torch; much larger image).
RUN . /venv/bin/activate && python -m spacy download en_core_web_lg

# git metadata for provenance (tira's runtime stats look for a repo at ./)
ADD .git /auto-judge/.git

