# libraries --------------------------------------------------------------------
library(tidyverse)
library(cowplot)
library(ggdist)
library(jsonlite)

dir.create("./figures", showWarnings = FALSE)

# style --------------------------------------------------------------------------
# categorical: control and CTNNB1 (validated palette, both groups also differ
# in shape); diverging (connectivity matrices): blue <-> red, gray midpoint at 0
COLORS <- c(control = "#2a78d6", test = "#eb6834")
SHAPES <- c(control = 16, test = 17)
LABELS <- c(control = "Control", test = "CTNNB1")
DIV_LOW <- "#1c5cab"
DIV_MID <- "#f0efec"
DIV_HIGH <- "#c43c3b"
INK <- "#0b0b0b"
MUTED <- "#52514e"

theme_paper <- function() {
  theme_minimal(base_size = 9) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "#e1e0d9", linewidth = 0.3),
      axis.text = element_text(color = MUTED),
      axis.title = element_text(color = INK),
      plot.title = element_text(color = INK, face = "bold", size = 9),
      legend.title = element_blank(),
      legend.text = element_text(color = INK),
      plot.background = element_rect(fill = "white", color = NA)
    )
}

# save as vector PDF and 600-dpi TIFF ------------------------------------------------
# plot_grid() sizes the text when it combines panels; size it with ragg (the
# same fonts as the saved figures) so that long labels get the room they need
set_null_device("agg")
save_figure <- function(plot, name, width, height) {
  ggsave(
    paste0("./figures/", name, ".pdf"), plot, width = width, height = height, units = "in",
    device = cairo_pdf, bg = "white"
  )
  ggsave(
    paste0("./figures/", name, ".tiff"), plot, width = width, height = height, units = "in",
    dpi = 600, device = ragg::agg_tiff, compression = "lzw", bg = "white"
  )
}

LEFT <- c("Fp1", "F7", "F3", "T3", "C3", "T5", "P3", "O1")
RIGHT <- c("Fp2", "F8", "F4", "T4", "C4", "T6", "P4", "O2")
CHANNELS <- c(
  "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
  "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"
)

# data ------------------------------------------------------------------------------------
model1 <- read_csv("./results/model1.csv", show_col_types = FALSE)
differences <- read_csv("./results/model1_differences.csv", show_col_types = FALSE)
model2_lines <- read_csv("./results/model2_lines.csv", show_col_types = FALSE)
sensitivity <- read_csv("./results/sensitivity_model1.csv", show_col_types = FALSE)
matched_set <- read_csv("./results/check_matched_set.csv", show_col_types = FALSE)
alpha <- read_csv("./data/connectome_metrics_alpha.csv", show_col_types = FALSE)
qc <- read_csv("./data/qc_summary.csv", show_col_types = FALSE)
pdr <- read_csv("./data/pdr.csv", show_col_types = FALSE)
norms <- fromJSON("./data/pdr_norms.json")

# Figure 3: interhemispheric connectivity in the Model 1 matched sample -----------------
# B: mean of the matched CTNNB1 children's matrices; A: mean over the same children
# of their matched controls' mean matrix; C: B minus A (the primary comparison)
interhemispheric <- function(id, group) {
  m <- as.matrix(read_csv(
    paste0("./data/connectomes/", group, "/alpha/", id, "_alpha_task-rest_connectome_eeg.csv"),
    col_names = FALSE, show_col_types = FALSE
  ))
  dimnames(m) <- list(CHANNELS, CHANNELS)
  m[LEFT, RIGHT]
}
mean_matrix <- function(ids, group) Reduce(`+`, map(ids, interhemispheric, group = group)) / length(ids)

matched <- differences %>% filter(analysis == "primary")
controls <- alpha %>% filter(group == "control")
test_matrix <- mean_matrix(matched$id, "test")
control_matrix <- Reduce(`+`, map2(matched$sex, matched$age, function(s, a) {
  mean_matrix(controls$id[controls$sex == s & controls$age == a], "control")
})) / nrow(matched)

as_long <- function(m) {
  as_tibble(m, rownames = "left") %>%
    pivot_longer(-left, names_to = "right", values_to = "value") %>%
    mutate(left = factor(left, levels = rev(LEFT)), right = factor(right, levels = RIGHT))
}
control_m <- as_long(control_matrix)
test_m <- as_long(test_matrix)
diff_m <- as_long(test_matrix - control_matrix)

matrix_panel <- function(df, title, fill_scale) {
  ggplot(df, aes(right, left, fill = value)) +
    geom_tile(color = "white", linewidth = 0.4) +
    fill_scale +
    coord_equal() +
    labs(title = title, x = "Right hemisphere", y = "Left hemisphere") +
    theme_paper() +
    theme(
      plot.title.position = "plot",
      panel.grid.major = element_blank(),
      axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
      legend.title = element_text(color = INK, size = 8),
      legend.key.height = unit(0.25, "in")
    )
}

# diverging scales centred on 0, so negative values stay visible; A and B share
# one scale, C is symmetric
level_scale <- scale_fill_gradient2(
  low = DIV_LOW, mid = DIV_MID, high = DIV_HIGH, midpoint = 0,
  limits = range(0, control_m$value, test_m$value), name = "dbWPLI"
)
div_limit <- max(abs(diff_m$value))
div_scale <- scale_fill_gradient2(
  low = DIV_LOW, mid = DIV_MID, high = DIV_HIGH, midpoint = 0,
  limits = c(-div_limit, div_limit), name = "Difference"
)

fig3 <- plot_grid(
  matrix_panel(control_m, "A  Matched controls", level_scale),
  matrix_panel(test_m, "B  CTNNB1", level_scale),
  matrix_panel(diff_m, "C  CTNNB1 − controls", div_scale),
  nrow = 1
)
save_figure(fig3, "figure3_matrices", width = 7.5, height = 2.8)

# Figure 4: per-child differences and robustness ----------------------------------------------
primary <- model1 %>% filter(analysis == "primary")
primary_d <- differences %>% filter(analysis == "primary")

fig4a <- ggplot(primary_d, aes(x = age, y = raw)) +
  annotate(
    "rect", xmin = -Inf, xmax = Inf, ymin = primary$raw_lower, ymax = primary$raw_upper,
    fill = COLORS["test"], alpha = 0.15
  ) +
  geom_hline(yintercept = 0, color = MUTED, linewidth = 0.4) +
  geom_hline(yintercept = primary$raw, color = COLORS["test"], linewidth = 0.6) +
  geom_point(
    color = COLORS["test"], shape = SHAPES["test"], size = 2,
    position = position_jitter(width = 0.15, height = 0, seed = 42)
  ) +
  scale_x_continuous(breaks = 0:9) +
  labs(
    title = "A",
    x = "Age (years)",
    y = "IHS difference\n(CTNNB1 minus matched controls)"
  ) +
  theme_paper()

robustness <- bind_rows(
  model1 %>% filter(analysis %in% c("primary", "S1 individual band", "S6 theta")),
  sensitivity,
  matched_set %>% transmute(
    analysis, mu = mean, mu_lower = lower, mu_upper = upper, n_test = primary$n_test, n_control = primary$n_control
  ),
  # post hoc checks of the competing explanations (analysis_posthoc.R)
  read_csv("./results/posthoc.csv", show_col_types = FALSE) %>%
    filter(analysis %in% c(
      "post hoc: overlapping alpha power, adjusted", "post hoc: control alpha power degraded with noise",
      "post hoc: blinks near the analysed epochs", "post hoc: epochs matched for alpha power"
    )) %>%
    select(analysis, mu, mu_lower, mu_upper, n_test, n_control)
) %>%
  mutate(
    label = recode(
      analysis,
      "primary" = "Primary (8-13 Hz)",
      "S1 individual band" = "Individual band (peak ± 2 Hz)",
      "S2 age 0-4" = "Age 2-4 years",
      "S2 age 5+" = "Age 5-9 years",
      "S3 without slowing" = "Without weak or slow alpha peak",
      "S4 without sleep" = "Without signs of sleep",
      "S8a data quality" = "Adjusted for data quality",
      "S8b data quality and alpha power" = "Adjusted for quality and alpha power",
      "S9 without Autoreject mis-calibration" = "Without miscalibrated cleaning",
      "S10 without interpolated electrodes" = "Without interpolated electrodes",
      "S11 without frontal electrodes" = "Without Fp1, Fp2, F7 and F8",
      "prior mu ~ N(0, 0.5)" = "Prior N(0, 0.5)",
      "prior mu ~ N(0, 2)" = "Prior N(0, 2)",
      "S6 theta" = "Theta (4-7 Hz)",
      "check: matched-set model" = "Matched-set model",
      "post hoc: overlapping alpha power, adjusted" = "Comparable alpha power, adjusted †",
      "post hoc: control alpha power degraded with noise" = "Noise added to control recordings †",
      "post hoc: blinks near the analysed epochs" = "Blinks near most epochs †",
      "post hoc: epochs matched for alpha power" = "Epochs matched for alpha power †"
    ),
    label = if_else(is.na(n_test), label, paste0(label, " (n = ", n_test, "/", n_control, ")")),
    # the checks that defined robustness
    label = if_else(analysis %in% c(
      "S1 individual band", "S2 age 0-4", "S2 age 5+", "S3 without slowing", "S4 without sleep",
      "S11 without frontal electrodes"
    ), paste0(label, " *"), label),
    label = fct_rev(fct_inorder(label))
  )

fig4b <- ggplot(robustness, aes(y = label, x = mu, xmin = mu_lower, xmax = mu_upper)) +
  geom_vline(xintercept = 0, color = MUTED, linewidth = 0.4) +
  geom_pointinterval(color = INK, point_size = 1.8, interval_size = 0.8) +
  labs(title = "B", x = "Standardised difference\n(95% credible interval)", y = NULL) +
  theme_paper()

fig4 <- plot_grid(fig4a, fig4b, nrow = 1, rel_widths = c(1, 1.25))
save_figure(fig4, "figure4_differences", width = 7.5, height = 3.2)

# Figure 5: interhemispheric strength and age (Model 2 sample) ----------------------------------
# 95% bands; each group's line spans its observed ages, so together the lines
# span the ages of the Model 2 sample (analysis_age.R)
lines_ihs <- model2_lines %>% filter(metric == "lh_rh")
ages <- seq(min(lines_ihs$age), max(lines_ihs$age))
young <- alpha %>% filter(age >= min(ages), age <= max(ages))

fig5 <- ggplot() +
  geom_ribbon(data = lines_ihs, aes(x = age, ymin = lower, ymax = upper, fill = group), alpha = 0.15) +
  geom_line(data = lines_ihs, aes(x = age, y = mean, color = group), linewidth = 0.7) +
  geom_point(
    data = young, aes(x = age, y = lh_rh, color = group, shape = group), size = 1.8, alpha = 0.8,
    position = position_jitter(width = 0.15, height = 0, seed = 42)
  ) +
  scale_color_manual(values = COLORS, labels = LABELS) +
  scale_fill_manual(values = COLORS, labels = LABELS) +
  scale_shape_manual(values = SHAPES, labels = LABELS) +
  scale_x_continuous(breaks = ages) +
  labs(x = "Age (years)", y = "Interhemispheric strength (dbWPLI)") +
  theme_paper() +
  theme(legend.position = "top")
save_figure(fig5, "figure5_age", width = 5, height = 3.5)

# Figure 6: cohort flow -------------------------------------------------------------------------
n_rec <- qc %>% count(group, name = "n")
n_excl <- qc %>% filter(excluded) %>% count(group, name = "n")
n_an <- qc %>% filter(!excluded) %>% count(group, name = "n")
n_matched <- differences %>% filter(analysis == "primary") %>% nrow()
n_matched_c <- primary$n_control

excluded_text <- function(g) {
  ex <- qc %>% filter(excluded, group == g)
  if (nrow(ex) == 0) return("Excluded (n = 0)")
  reasons <- ex %>%
    mutate(reason = if_else(str_detect(exclusion_reason, "channels"), "more than three\nnon-working electrodes", "too few epochs free\nof large artefacts")) %>%
    count(reason)
  counts <- if (nrow(reasons) > 1) paste0(" (n = ", reasons$n, ")") else ""
  paste0("Excluded (n = ", nrow(ex), "):\n", paste0(reasons$reason, counts, collapse = "\n"))
}

get_n <- function(df, g) {
  x <- df %>% filter(group == g) %>% pull(n)
  if (length(x) == 0) 0 else x
}

boxes <- tribble(
  ~x, ~y, ~label,
  1, 4, paste0("CTNNB1 syndrome: recordings\nat 2-9 years (n = ", get_n(n_rec, "test"), ")"),
  1, 2.9, excluded_text("test"),
  1, 1.8, paste0("Analysable (n = ", get_n(n_an, "test"), ")"),
  1, 0.8, paste0("Matched by sex and age\n(n = ", n_matched, ")"),
  3, 4, paste0("Controls: normal clinical EEGs\nat 2-9 years (n = ", get_n(n_rec, "control"), ")"),
  3, 2.9, excluded_text("control"),
  3, 1.8, paste0("Analysable (n = ", get_n(n_an, "control"), ")"),
  3, 0.8, paste0("Matched controls\n(n = ", n_matched_c, ")")
)
arrows <- tribble(
  ~x, ~y, ~yend,
  1, 3.75, 3.3, 1, 2.5, 2.05, 1, 1.55, 1.1,
  3, 3.75, 3.3, 3, 2.5, 2.05, 3, 1.55, 1.1
)

fig6 <- ggplot() +
  geom_segment(
    data = arrows, aes(x = x, xend = x, y = y, yend = yend),
    arrow = arrow(length = unit(0.06, "in"), type = "closed"), color = MUTED, linewidth = 0.4
  ) +
  geom_label(
    data = boxes, aes(x = x, y = y, label = label), size = 2.6, color = INK,
    label.size = 0.3, label.padding = unit(0.15, "in"), fill = "white", lineheight = 0.95
  ) +
  xlim(0, 4) +
  ylim(0.3, 4.5) +
  theme_void() +
  theme(plot.background = element_rect(fill = "white", color = NA))
save_figure(fig6, "figure6_flow", width = 6, height = 3.6)

# Supplementary figure: PDR and age ---------------------------------------------------------------
age_grid <- tibble(age = seq(min(pdr$age), max(pdr$age), by = 0.1)) %>%
  mutate(mean = norms$pdr$params$const + norms$pdr$params$age * log(age + 1))

figS1 <- ggplot(pdr %>% filter(pdr_clear), aes(x = age, y = pdr_hz)) +
  geom_line(data = age_grid, aes(x = age, y = mean), color = MUTED, linewidth = 0.6) +
  geom_point(aes(color = group, shape = group), size = 1.8, alpha = 0.8, position = position_jitter(width = 0.15, height = 0, seed = 42)) +
  geom_hline(yintercept = c(8, 13), color = MUTED, linetype = "dashed", linewidth = 0.3) +
  scale_color_manual(values = COLORS, labels = LABELS) +
  scale_shape_manual(values = SHAPES, labels = LABELS) +
  labs(x = "Age (years)", y = "Posterior dominant rhythm (Hz)") +
  theme_paper() +
  theme(legend.position = "top")
save_figure(figS1, "figureS1_pdr", width = 5, height = 3.5)

# Supplementary figure: interhemispheric strength and alpha power (post hoc) --------------------
# shaded: the range of posterior alpha power observed in both groups (analysis_posthoc.R)
posthoc <- read_csv("./results/posthoc.csv", show_col_types = FALSE)
alpha_range <- posthoc %>% filter(analysis == "post hoc: overlapping alpha power, adjusted")
alpha_power <- alpha %>% left_join(pdr %>% select(id, alpha_adj_db), by = "id")

figS3 <- ggplot(alpha_power, aes(x = alpha_adj_db, y = lh_rh)) +
  annotate(
    "rect", xmin = alpha_range$alpha_low, xmax = alpha_range$alpha_high, ymin = -Inf, ymax = Inf,
    fill = MUTED, alpha = 0.12
  ) +
  geom_point(aes(color = group, shape = group), size = 1.8, alpha = 0.8) +
  scale_color_manual(values = COLORS, labels = LABELS) +
  scale_shape_manual(values = SHAPES, labels = LABELS) +
  labs(x = "Posterior alpha power (dB above the aperiodic component)", y = "Interhemispheric strength (dbWPLI)") +
  theme_paper() +
  theme(legend.position = "top")
save_figure(figS3, "figureS3_alpha", width = 5, height = 3.5)

# Supplementary figure: posterior power and IHS frequency by frequency (post hoc) ---------------
# group means with 95% intervals (mean +/- 1.96 SE) over all analysable children
by_freq <- read_csv("./data/ihs_by_frequency.csv", show_col_types = FALSE) %>%
  pivot_longer(c(power_db, ihs), names_to = "measure", values_to = "value") %>%
  group_by(group, freq, measure) %>%
  summarise(mean = mean(value), se = sd(value) / sqrt(n()), .groups = "drop") %>%
  mutate(lower = mean - 1.96 * se, upper = mean + 1.96 * se)

freq_panel <- function(m, title, ylab) {
  ggplot(by_freq %>% filter(measure == m), aes(x = freq, y = mean, color = group, fill = group)) +
    annotate("rect", xmin = 8, xmax = 13, ymin = -Inf, ymax = Inf, fill = MUTED, alpha = 0.08) +
    geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.15, color = NA) +
    geom_line(linewidth = 0.7) +
    scale_color_manual(values = COLORS, labels = LABELS) +
    scale_fill_manual(values = COLORS, labels = LABELS) +
    labs(title = title, x = "Frequency (Hz)", y = ylab) +
    theme_paper() +
    theme(legend.position = "top")
}
figS4 <- plot_grid(
  freq_panel("power_db", "A", "Posterior power (dB)"),
  freq_panel("ihs", "B", "Interhemispheric strength (dbWPLI)"),
  nrow = 1
)
save_figure(figS4, "figureS4_frequency", width = 7.5, height = 3.2)
