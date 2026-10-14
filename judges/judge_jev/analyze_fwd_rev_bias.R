#!/usr/bin/env Rscript
#
# analyze_fwd_rev_bias.R -- positional-bias analysis for JudgeJev's raw JEV capture.
#
# Reads the `.qrels_jev_raw.jsonl` sidecar written by JevQrelsCreator (judge_jev.py):
# one row per unique cited document, with the untouched answers to all 5 relevance
# questions -- 1 Noul, 2 Choice (forward/reverse option order), 2 Score (forward/
# reverse scale order). This script asks: did presenting the SAME TREC 0-3 grade
# scale in a different order change what JEV said? That's positional/order bias.
#
# For each document we reduce every answer to a "grade" on the common 0-3 scale:
#   - noul:        grade = noul * GRADE_MAX                      (no fwd/rev pair)
#   - choice_fwd/rev: grade = sum(option_value * probability)      (probabilities are
#     keyed by the actual grade label regardless of presentation order -- no
#     inversion needed to compare fwd vs rev on the same scale)
#   - score_fwd:   grade = score                                  (array index == grade)
#   - score_rev:   grade = GRADE_MAX - score                      (array was reversed,
#     so the raw index needs inverting before it's comparable to score_fwd)
#
# This mirrors judge_jev.py's `_signal_grade` / `combine_to_probability` exactly.
# GRADE_MAX is NOT read from the Python module (no shared source of truth across
# languages) -- keep it in sync with judge_jev.py's GRADE_MAX by hand if that
# changes.
#
# Usage:
#   Rscript analyze_fwd_rev_bias.R <path/to/judge_jev.qrels_jev_raw.jsonl> [output_dir]
#
# Outputs (under output_dir, default "<input_dir>/bias_analysis/"):
#   graded_long.csv, graded_wide.csv   -- tidy processed data, one row per
#                                         (document, question_type) pair for _wide
#   bias_summary.csv                   -- per-question-type bias stats
#   fwd_vs_rev_scatter.png
#   bland_altman.png
#   diff_distribution.png
#   fwd_vs_rev_boxplot.png

suppressPackageStartupMessages({
  library(tidyverse)
  library(jsonlite)
})

GRADE_MAX <- 3L

# ---- CLI args -------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript analyze_fwd_rev_bias.R <path/to/*.qrels_jev_raw.jsonl> [output_dir]",
       call. = FALSE)
}


input_path <- args[[1]]
input_path <- '~/Repos/auto-judge-starter-kit/output-rag26/judge_jev.qrels_jev_raw.jsonl'
output_dir <- if (length(args) >= 2) args[[2]] else file.path(dirname(input_path), "bias_analysis")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(input_path)) {
  stop(sprintf("Input file not found: %s", input_path), call. = FALSE)
}

# ---- Parse the raw JSONL ----------------------------------------------------
#
# One JSON object per line; `answers` is a nested object (or JSON null if the JEV
# call failed outright for that document) keyed by the 5 question names.

lines <- read_lines(input_path)
message(sprintf("Read %d rows from %s", length(lines), input_path))

parse_row <- function(line) {
  rec <- fromJSON(line, simplifyVector = FALSE)
  if (is.null(rec$answers)) {
    return(tibble(
      topic_id = rec$topic_id, shard = rec$shard, doc_id = rec$doc_id,
      failed = TRUE,
      noul_grade = NA_real_,
      choice_fwd_grade = NA_real_, choice_rev_grade = NA_real_,
      choice_fwd_confidence = NA_real_, choice_rev_confidence = NA_real_,
      score_fwd_grade = NA_real_, score_rev_grade = NA_real_,
      score_fwd_confidence = NA_real_, score_rev_confidence = NA_real_,
    ))
  }

  a <- rec$answers
  choice_grade <- function(ans) {
    probs <- ans$probabilities
    if (is.null(probs) || length(probs) == 0) return(NA_real_)
    sum(as.numeric(names(probs)) * as.numeric(unlist(probs)))
  }

  tibble(
    topic_id = rec$topic_id, shard = rec$shard, doc_id = rec$doc_id,
    failed = FALSE,
    noul_grade = unname(a$relevant_noul$noul) * GRADE_MAX,
    choice_fwd_grade = choice_grade(a$relevant_choice_fwd),
    choice_rev_grade = choice_grade(a$relevant_choice_rev),
    choice_fwd_confidence = a$relevant_choice_fwd$confidence %||% NA_real_,
    choice_rev_confidence = a$relevant_choice_rev$confidence %||% NA_real_,
    score_fwd_grade = a$relevant_score_fwd$score %||% NA_real_,
    score_rev_grade = GRADE_MAX - (a$relevant_score_rev$score %||% NA_real_),
    score_fwd_confidence = a$relevant_score_fwd$confidence %||% NA_real_,
    score_rev_confidence = a$relevant_score_rev$confidence %||% NA_real_,
  )
}

graded <- map_dfr(lines, parse_row)

n_failed <- sum(graded$failed)
if (n_failed > 0) {
  message(sprintf("Dropping %d/%d documents with no JEV answer (call failed)", n_failed, nrow(graded)))
}
graded <- filter(graded, !failed) |> select(-failed)

# ---- Reshape into paired fwd/rev long + wide format -------------------------
#
# graded_long: one row per (document, question_type, direction) -- for the
#   fwd-vs-rev boxplot.
# graded_wide: one row per (document, question_type) with `fwd`/`rev` columns --
#   for the scatter, Bland-Altman and paired significance tests.

graded_long <- graded |>
  select(topic_id, shard, doc_id,
         choice_fwd = choice_fwd_grade, choice_rev = choice_rev_grade,
         score_fwd = score_fwd_grade, score_rev = score_rev_grade) |>
  pivot_longer(
    cols = c(choice_fwd, choice_rev, score_fwd, score_rev),
    names_to = c("question_type", "direction"),
    names_sep = "_",
    values_to = "grade"
  )

graded_wide <- graded_long |>
  pivot_wider(names_from = direction, values_from = grade) |>
  mutate(
    diff = rev - fwd,          # positive: reverse presentation graded it HIGHER
    mean_grade = (fwd + rev) / 2,
  ) |>
  drop_na(fwd, rev)

write_csv(graded_long, file.path(output_dir, "graded_long.csv"))
write_csv(graded_wide, file.path(output_dir, "graded_wide.csv"))

# ---- Bias summary: is the fwd/rev difference systematically non-zero? ------

bias_summary <- graded_wide |>
  group_by(question_type) |>
  summarise(
    n = n(),
    mean_fwd = mean(fwd), mean_rev = mean(rev),
    mean_diff = mean(diff), sd_diff = sd(diff),
    correlation = cor(fwd, rev),
    t_statistic = t.test(diff)$statistic,
    p_value_t = t.test(diff)$p.value,
    p_value_wilcoxon = suppressWarnings(wilcox.test(diff)$p.value),
    .groups = "drop"
  ) |>
  mutate(significant_bias_a0.05 = p_value_t < 0.05)

write_csv(bias_summary, file.path(output_dir, "bias_summary.csv"))

message("\n=== Positional-bias summary (reverse - forward, on the 0-", GRADE_MAX, " grade scale) ===")
print(bias_summary, width = Inf)

# ---- Plots -------------------------------------------------------------------

theme_set(theme_minimal(base_size = 12))

# 1. fwd vs rev scatter, with the identity line -- points off the diagonal are
#    where presentation order changed JEV's answer.
p_scatter <- ggplot(graded_wide, aes(x = fwd, y = rev, color = topic_id)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey40") +
  geom_point(alpha = 0.7, size = 2) +
  facet_wrap(~question_type) +
  coord_equal(xlim = c(0, GRADE_MAX), ylim = c(0, GRADE_MAX)) +
  labs(
    title = "Forward vs. reverse presentation order",
    subtitle = "Points on the dashed line agree regardless of option/scale order",
    x = "Grade (forward order)", y = "Grade (reverse order)", color = "Topic"
  )
ggsave(file.path(output_dir, "fwd_vs_rev_scatter.png"), p_scatter, width = 9, height = 5, dpi = 150)

# 2. Bland-Altman: mean grade vs. difference, with limits of agreement.
#    A non-zero mean difference line is the headline positional-bias signal.
loa <- bias_summary |>
  mutate(upper = mean_diff + 1.96 * sd_diff, lower = mean_diff - 1.96 * sd_diff)

p_ba <- ggplot(graded_wide, aes(x = mean_grade, y = diff)) +
  geom_point(alpha = 0.6, color = "steelblue") +
  geom_hline(yintercept = 0, linetype = "dotted", color = "grey50") +
  geom_hline(data = loa, aes(yintercept = mean_diff), color = "firebrick") +
  geom_hline(data = loa, aes(yintercept = upper), linetype = "dashed", color = "firebrick") +
  geom_hline(data = loa, aes(yintercept = lower), linetype = "dashed", color = "firebrick") +
  facet_wrap(~question_type) +
  labs(
    title = "Bland-Altman: agreement between forward and reverse order",
    subtitle = "Solid red = mean bias; dashed red = 95% limits of agreement (±1.96 SD)",
    x = "Mean of forward & reverse grade", y = "Difference (reverse - forward)"
  )
ggsave(file.path(output_dir, "bland_altman.png"), p_ba, width = 9, height = 5, dpi = 150)

# 3. Distribution of the paired difference -- shape/skew of the bias, not just its mean.
p_diff <- ggplot(graded_wide, aes(x = diff, fill = question_type)) +
  geom_histogram(binwidth = 0.25, alpha = 0.8, position = "identity") +
  geom_vline(xintercept = 0, linetype = "dotted", color = "black") +
  geom_vline(data = bias_summary, aes(xintercept = mean_diff), color = "firebrick", linewidth = 0.8) +
  facet_wrap(~question_type, scales = "free_y") +
  labs(
    title = "Distribution of reverse - forward difference",
    subtitle = "Dotted = zero bias; solid red = observed mean bias",
    x = "Difference (reverse - forward)", y = "Count"
  ) +
  theme(legend.position = "none")
ggsave(file.path(output_dir, "diff_distribution.png"), p_diff, width = 9, height = 5, dpi = 150)

# 4. Paired boxplot of fwd vs rev per question type -- a second view of the same
#    shift, without the pairing (useful alongside the scatter/Bland-Altman).
p_box <- ggplot(graded_long, aes(x = direction, y = grade, fill = direction)) +
  geom_boxplot(alpha = 0.7, outlier.alpha = 0.4) +
  facet_wrap(~question_type) +
  labs(title = "Forward vs. reverse grade distribution", x = NULL, y = "Grade") +
  theme(legend.position = "none")
ggsave(file.path(output_dir, "fwd_vs_rev_boxplot.png"), p_box, width = 7, height = 5, dpi = 150)

message(sprintf("\nWrote plots and tidy CSVs to %s", normalizePath(output_dir)))
