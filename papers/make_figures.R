# ggplot2 versions of the paper figures, in the co-author's style (theme_minimal, cream
# background, no grid, translucent dark points; accents steelblue / firebrick / grey40).
#
#   python papers/make_figures.py      # writes the tidy CSVs in papers/figures/data/
#   Rscript papers/make_figures.R      # writes papers/figures_ggplot/*.pdf and *.png
#
# Plots exactly the data the seaborn figures plot (window topics only; F11 uses the
# submitted all-topic leaderboards). CIs here are mean +/- 1.96 SE (seaborn bootstraps),
# except F7, which bootstraps over topics as in the paper.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(tidyr)
  library(patchwork)
})

args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
data_dir <- file.path(here, "figures", "data")
out_dir <- file.path(here, "figures_ggplot")
dir.create(out_dir, showWarnings = FALSE)

cream <- "#FFFBF0"
judge_cols <- c("Jev Choice" = "steelblue", "Jev mirror" = "firebrick",
                "Gemini" = "grey40", "BOD" = "darkgoldenrod", "BOD_GROUNDED" = "darkolivegreen4")
ds_cols <- c(rag26 = "steelblue", ragtime26 = "firebrick")

theme_set(
  theme_minimal(base_size = 10) +
    theme(
      plot.background = element_rect(fill = cream, colour = NA),
      panel.background = element_rect(fill = cream, colour = NA),
      legend.background = element_rect(fill = cream, colour = NA),
      panel.grid = element_blank(),
      axis.text = element_text(colour = "grey40", size = 8),
      axis.title = element_text(colour = "grey30", size = 9),
      axis.ticks = element_line(colour = "grey70", linewidth = .3),
      axis.line = element_line(colour = "grey80", linewidth = .3),
      strip.text = element_text(colour = "grey20", size = 9, face = "bold"),
      plot.title = element_text(colour = "grey20", size = 10),
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text = element_text(size = 8)
    )
)

rd <- function(name) {
  d <- read_csv(file.path(data_dir, paste0(name, ".csv")), show_col_types = FALSE)
  if ("judge" %in% names(d)) d$judge <- factor(d$judge, levels = intersect(names(judge_cols), unique(d$judge)))
  d
}

ci <- function(df, value, ...) {
  df |>
    group_by(...) |>
    summarise(mean = mean({{ value }}), se = sd({{ value }}) / sqrt(n()), n = n(), .groups = "drop") |>
    mutate(lo = mean - 1.96 * se, hi = mean + 1.96 * se)
}

save <- function(p, name, w, h) {
  for (ext in c("pdf", "png")) {
    dev <- if (ext == "pdf") cairo_pdf else "png"
    ggsave(file.path(out_dir, paste0(name, ".", ext)), p, width = w, height = h, dpi = 200,
           device = dev, bg = cream)
  }
  message("[", name, "] written")
}

ref_line <- function(y) geom_hline(yintercept = y, linetype = "dotted", colour = "grey50", linewidth = .4)

# F1 ------------------------------------------------------------------------------------
f01 <- rd("f01_probability_distribution") |> rename(p = `P(A better)`)
p <- ggplot(f01, aes(p, colour = judge, fill = judge)) +
  geom_histogram(aes(y = after_stat(width * density)), binwidth = .025, boundary = 0,
                 position = "identity", alpha = .25, linewidth = .3) +
  facet_wrap(~dataset, scales = "free_y") +
  scale_colour_manual(values = judge_cols) + scale_fill_manual(values = judge_cols) +
  labs(x = "P(A better)", y = "share of comparisons")
save(p, "f01_probability_distribution", 7.5, 2.9)

# F2 ------------------------------------------------------------------------------------
f02 <- rd("f02_position_bias") |> rename(pa = `P(slot A wins)`)
s <- ci(f02, pa, dataset, judge, bin, bin_label)
p <- ggplot(s, aes(reorder(bin_label, bin), mean, colour = judge, group = judge)) +
  ref_line(.5) +
  geom_line(linewidth = .5, alpha = .7) +
  geom_pointrange(aes(ymin = lo, ymax = hi), size = .25, position = position_dodge(.25)) +
  facet_wrap(~dataset) +
  scale_colour_manual(values = judge_cols) +
  labs(x = "orientation-averaged P(x better)", y = "P(slot A wins)") +
  theme(axis.text.x = element_text(angle = 40, hjust = 1))
save(p, "f02_position_bias", 7.5, 3.1)

# F3 ------------------------------------------------------------------------------------
f03 <- rd("f03_identical_summaries") |> rename(p = `P(A better)`)
p <- ggplot(f03, aes(judge, p, colour = judge)) +
  ref_line(.5) +
  geom_jitter(width = .25, height = 0, alpha = .2, size = .6) +
  stat_summary(fun = mean, geom = "crossbar", width = .6, linewidth = .4, colour = "black") +
  facet_wrap(~dataset) +
  scale_colour_manual(values = judge_cols, guide = "none") +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = NULL, y = "P(A better), identical summaries (bar = mean)")
save(p, "f03_identical_summaries", 5.2, 3)

# F4 ------------------------------------------------------------------------------------
f04a <- rd("f04_length_ratio") |> rename(pl = `P(longer wins)`, ratio = `length ratio`)
s <- ci(f04a, pl, dataset, judge, ratio) |>
  mutate(ratio = factor(ratio, levels = c("<1.25×", "1.25–2×", "≥2×")))
pa <- ggplot(s, aes(ratio, mean, colour = judge, group = judge)) +
  ref_line(.5) +
  geom_line(linewidth = .5, alpha = .7) +
  geom_pointrange(aes(ymin = lo, ymax = hi), size = .25, position = position_dodge(.2)) +
  facet_wrap(~dataset) +
  scale_colour_manual(values = judge_cols) +
  labs(x = "length ratio (longer / shorter)", y = "P(longer wins)")
f04b <- rd("f04_length_rho") |> rename(rho = `Spearman(score, words)`) |>
  mutate(judge = factor(judge, levels = names(judge_cols)[1:4]))
pb <- ggplot(f04b, aes(judge, rho, colour = judge)) +
  ref_line(0) +
  geom_jitter(width = .15, height = 0, alpha = .6, size = 1) +
  stat_summary(fun = median, geom = "crossbar", width = .5, linewidth = .4, colour = "black") +
  facet_wrap(~dataset) +
  scale_colour_manual(values = judge_cols, guide = "none") +
  labs(x = NULL, y = "per-topic Spearman(score, words)\n(bar = median)") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))
save(pa + pb + plot_layout(widths = c(1.2, 1)), "f04_length", 10, 3.2)

# F5 ------------------------------------------------------------------------------------
qs <- c("better_summary", "length_neutral", "needs_coverage", "precision")
f05 <- rd("f05_prompt_agreement") |>
  mutate(row = factor(gsub("_", " ", row), levels = rev(gsub("_", " ", qs))),
         col = factor(gsub("_", " ", col), levels = gsub("_", " ", qs)))
p <- ggplot(f05, aes(col, row, fill = rho)) +
  geom_tile(colour = cream, linewidth = .8) +
  geom_text(aes(label = sprintf("%.3f", rho)), size = 2.8, colour = "grey10") +
  facet_wrap(~dataset) +
  scale_fill_gradient(low = cream, high = "steelblue", limits = c(.8, 1), name = "mean per-topic ρ") +
  labs(x = NULL, y = NULL) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1), legend.title = element_text(size = 8)) +
  coord_fixed()
save(p, "f05_prompt_agreement", 7.5, 3.6)

# F6 ------------------------------------------------------------------------------------
var_order <- c("pad_repeat", "pad_generic", "pad_offtopic", "pad_relevant", "truncate")
f06 <- rd("f06_padding_pairwise") |> rename(d = `Δ`)
s <- ci(f06, d, dataset, judge, variant) |> mutate(variant = factor(variant, levels = var_order))
pa <- ggplot(s, aes(variant, mean, colour = judge)) +
  ref_line(0) +
  geom_pointrange(aes(ymin = lo, ymax = hi), size = .3, position = position_dodge(.4)) +
  facet_wrap(~dataset, labeller = labeller(dataset = c(rag26 = "rag26 (Jev 10 topics, Gemini 3)",
                                                       ragtime26 = "ragtime26 (Jev 10 topics, Gemini 3)"))) +
  scale_colour_manual(values = judge_cols) +
  labs(x = NULL, y = "ΔP(target wins) vs original") +
  theme(axis.text.x = element_text(angle = 35, hjust = 1))
f06b <- rd("f06_padding_bod") |> rename(d = `Δ`)
s <- ci(f06b, d, measure, variant) |> mutate(variant = factor(variant, levels = var_order))
pb <- ggplot(s, aes(variant, mean, colour = measure)) +
  ref_line(0) +
  geom_pointrange(aes(ymin = lo, ymax = hi), size = .3, position = position_dodge(.4)) +
  scale_colour_manual(values = c("ΔBOD_MEAN" = "darkgoldenrod", "ΔBOD_PADDING" = "grey40")) +
  labs(x = NULL, y = "Δ vs original", title = "rag26: BOD (graded rubric)") +
  theme(axis.text.x = element_text(angle = 35, hjust = 1))
save(pa + pb + plot_layout(widths = c(2, 1)), "f06_padding", 10.5, 3.4)

# F7 ------------------------------------------------------------------------------------
f07 <- rd("f07_broad_narrow")
set.seed(0)
boot <- f07 |>
  group_by(dataset) |>
  summarise(mean = mean(DiD),
            lo = quantile(replicate(4000, mean(sample(DiD, replace = TRUE))), .025),
            hi = quantile(replicate(4000, mean(sample(DiD, replace = TRUE))), .975), .groups = "drop")
p <- ggplot(f07, aes(dataset, DiD)) +
  ref_line(0) +
  geom_jitter(width = .08, height = 0, size = 1.6, alpha = .7, colour = "grey20") +
  geom_pointrange(data = boot, aes(y = mean, ymin = lo, ymax = hi), colour = "firebrick",
                  position = position_nudge(x = .3), size = .4, shape = 18) +
  labs(x = NULL, y = "Δ shift toward shorter\n(proportionate − better)",
       title = "narrow vs broad request, per window topic\n(red: mean, topic-bootstrap 95% CI)")
save(p, "f07_broad_narrow", 4.6, 3.2)

# F8 ------------------------------------------------------------------------------------
f08 <- rd("f08_judge_agreement") |>
  mutate(facet = sprintf("%s: %s, %d topics, ρ=%.3f", dataset,
                         ifelse(panel == "gemini", "Gemini", "BOD"), topics, rho),
         facet = factor(facet, levels = unique(facet)))
p <- ggplot(f08, aes(x, y, colour = ifelse(panel == "gemini", "Gemini", "BOD"))) +
  geom_point(alpha = .6, size = 1.1) +
  facet_wrap(~facet, nrow = 1, scales = "free") +
  scale_colour_manual(values = judge_cols, guide = "none") +
  labs(x = "Jev mirror win rate (window board)", y = "Gemini win rate / BOD_MEAN")
save(p, "f08_judge_agreement", 11, 3)

# F9 ------------------------------------------------------------------------------------
f09q <- rd("f09_bod_questions") |> rename(g = `most likely grade`, dg = `E[g] − most likely`)
share <- f09q |> count(dataset, g) |> group_by(dataset) |> mutate(share = n / sum(n))
pa <- ggplot(share, aes(factor(g), share, fill = dataset)) +
  geom_col(position = position_dodge(.8), width = .75, alpha = .85) +
  scale_fill_manual(values = ds_cols) +
  labs(x = "most likely grade", y = "share of questions")
pb <- ggplot(f09q, aes(dg, colour = dataset, fill = dataset)) +
  geom_histogram(aes(y = after_stat(width * density)), binwidth = 1 / 15, position = "identity",
                 alpha = .25, linewidth = .3) +
  scale_colour_manual(values = ds_cols) + scale_fill_manual(values = ds_cols) +
  labs(x = "expected − most likely grade", y = "share of questions")
pc <- ggplot(rd("f09_bod_answers"), aes(BOD_MEAN, colour = dataset, fill = dataset)) +
  geom_histogram(aes(y = after_stat(width * density)), binwidth = 1 / 30, boundary = 0,
                 position = "identity", alpha = .25, linewidth = .3) +
  scale_colour_manual(values = ds_cols) + scale_fill_manual(values = ds_cols) +
  labs(x = "BOD_MEAN per answer", y = "share of answers")
save((pa | pb | pc) + plot_layout(guides = "collect") & theme(legend.position = "bottom"),
     "f09_bod_grades", 10.5, 3.2)

# F10 -----------------------------------------------------------------------------------
f10 <- rd("f10_risk") |> rename(v = `share of answer concepts`)
pa <- ggplot(f10, aes(measure, v, colour = dataset)) +
  geom_jitter(position = position_jitterdodge(jitter.width = .25, dodge.width = .75),
              alpha = .15, size = .5) +
  geom_boxplot(fill = NA, outlier.shape = NA, width = .6, linewidth = .35,
               position = position_dodge(.75)) +
  scale_colour_manual(values = ds_cols) +
  labs(x = NULL, y = "share of answer concepts", title = "per answer (risk defined)")
pb <- ggplot(rd("f10_grounded_vs_mean"), aes(BOD_MEAN, BOD_GROUNDED)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey40", linewidth = .3) +
  geom_point(alpha = .2, size = .6, colour = "black") +
  facet_wrap(~dataset) +
  coord_equal() +
  labs(title = "per answer")
save((pa | pb) + plot_layout(widths = c(1, 2)), "f10_hallucination_risk", 10.5, 3.4)

# F11 -----------------------------------------------------------------------------------
f11 <- rd("f11_full_leaderboards") |>
  mutate(facet = sprintf("%s: %s vs %s\n%d runs, ρ=%.3f", dataset, yname, xname, runs, rho),
         facet = factor(facet, levels = unique(facet)))
p <- ggplot(f11, aes(x, y)) +
  geom_point(alpha = .6, size = 1.1, colour = "black") +
  facet_wrap(~facet, nrow = 2, scales = "free") +
  labs(x = "x-axis judge (see panel)", y = "y-axis judge (see panel)")
save(p, "f11_full_leaderboards", 10, 6)

# F12 -----------------------------------------------------------------------------------
f12 <- rd("f12_split_half") |> rename(r = `split-half Spearman`) |>
  mutate(judge = factor(judge, levels = c("Jev mirror", "BOD", "BOD_GROUNDED")))
p <- ggplot(f12, aes(judge, r, colour = judge)) +
  geom_jitter(width = .2, height = 0, alpha = .3, size = .6) +
  geom_boxplot(fill = NA, outlier.shape = NA, width = .5, linewidth = .35, colour = "black") +
  facet_wrap(~dataset) +
  scale_colour_manual(values = c(judge_cols, "BOD" = "darkgoldenrod"), guide = "none") +
  labs(x = NULL, y = "split-half Spearman (200 random 5/5 splits)")
save(p, "f12_split_half", 5.6, 3.2)

# F13 -----------------------------------------------------------------------------------
f13 <- rd("f13_determinism") |>
  mutate(facet = sprintf("%s (1 topic, n=%s): flips %.2f%%", dataset,
                         format(n, big.mark = ","), 100 * flips))
p <- ggplot(f13, aes(x, y)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey40", linewidth = .3) +
  geom_point(alpha = .15, size = .5, colour = "black") +
  facet_wrap(~facet) +
  coord_equal() +
  labs(x = "mirror P, request 1", y = "mirror P, request 2")
save(p, "f13_determinism", 6.4, 3.4)
