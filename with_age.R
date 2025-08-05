# libraries and utils ----------------------------------------------------------
library(tidyverse)
library(cowplot)

source("./utils/simple_linear.R")

# load data --------------------------------------------------------------------
band <- "alpha"
df_metrics <- read_csv(paste0("./data/connectome_metrics_", band, ".csv"))

# pairs ------------------------------------------------------------------------
age_difference <- 0

df_metrics_test <- df_metrics %>%
  filter(group == "test")
df_metrics_control <- df_metrics %>%
  filter(group == "control")

df_pairs <- df_metrics_test %>%
  inner_join(df_metrics_control, by = "sex", suffix = c("_test", "_control")) %>%
  filter(abs(age_test - age_control) <= age_difference) %>%
  transmute(
    id = id_test,
    id_control = id_control,
    lh_rh = lh_rh_test - lh_rh_control,
    laq_raq = laq_raq_test - laq_raq_control,
    lpq_rpq = lpq_rpq_test - lpq_rpq_control,
    laq_rpq = laq_rpq_test - laq_rpq_control,
    raq_lpq = raq_lpq_test - raq_lpq_control,
    ge = ge_test - ge_control,
    age_difference = abs(age_test - age_control)
  )

test_ids <- unique(df_pairs$id)
control_ids <- unique(df_pairs$id_control)

# subset df_metrics_test and df_metrics_control to only include ids in df_pairs
df_metrics_test <- df_metrics_test %>%
  filter(id %in% test_ids)
df_metrics_control <- df_metrics_control %>%
  filter(id %in% control_ids)


# ge ---------------------------------------------------------------------------
# plot storage
ge_plots <- list()

# metric
y_test_ge <- df_metrics_test$ge
y_control_ge <- df_metrics_control$ge

# test
df_test_ge <- data.frame(
  x = df_metrics_test$age,
  y = y_test_ge
)
fit_test_ge <- fit_simple_linear(
  x = df_test_ge$x,
  y = df_test_ge$y
)
compare_simple_linear(fit = fit_test_ge)

ge_plots[["test"]] <- plot_simple_linear(fit = fit_test_ge, data = df_test_ge) +
  ggtitle("Test group") +
  xlab("Age (years)") +
  ylab("Global efficiency (GE)") +
  ylim(-0.02, 0.02)

# control
df_control_ge <- data.frame(
  x = df_metrics_control$age,
  y = y_control_ge
)
fit_control_ge <- fit_simple_linear(
  x = df_control_ge$x,
  y = df_control_ge$y
)
compare_simple_linear(fit = fit_control_ge)
ge_plots[["control"]] <- plot_simple_linear(fit = fit_control_ge, data = df_control_ge) +
  ggtitle("Control group") +
  xlab("Age (years)") +
  ylab("Global efficiency (GE)") +
  ylim(-0.02, 0.02)

plot_grid(plotlist = ge_plots, ncol = 2, scale = 0.9)
ggsave(
  paste0("./figures/ge_age_", band, ".png"),
  width = 1920,
  height = 1080,
  dpi = 150,
  units = "px",
  bg = "white"
)

# lh_rh ------------------------------------------------------------------------
# plot storage
lh_rh_plots <- list()

# metric
y_test_lh_rh <- df_metrics_test$lh_rh
y_control_lh_rh <- df_metrics_control$lh_rh

# test
df_test_lh_rh <- data.frame(
  x = df_metrics_test$age,
  y = y_test_lh_rh
)
fit_test_lh_rh <- fit_simple_linear(
  x = df_test_lh_rh$x,
  y = df_test_lh_rh$y
)
compare_simple_linear(fit = fit_test_lh_rh)
lh_rh_plots[["test"]] <- plot_simple_linear(fit = fit_test_lh_rh, data = df_test_lh_rh) +
  ggtitle("Test group") +
  xlab("Age (years)") +
  ylab("Average interhemispheric strength (IHS)")

# control
df_control_lh_rh <- data.frame(
  x = df_metrics_control$age,
  y = y_control_lh_rh
)
fit_control_lh_rh <- fit_simple_linear(
  x = df_control_lh_rh$x,
  y = df_control_lh_rh$y
)
compare_simple_linear(fit = fit_control_lh_rh)
lh_rh_plots[["control"]] <- plot_simple_linear(fit = fit_control_lh_rh, data = df_control_lh_rh) +
  ggtitle("Control group") +
  xlab("Age (years)") +
  ylab("Average interhemispheric strength (IHS)")

plot_grid(plotlist = lh_rh_plots, ncol = 2, scale = 0.9)
ggsave(
  paste0("./figures/ihs_age_", band, ".png"),
  width = 1920,
  height = 1080,
  dpi = 150,
  units = "px",
  bg = "white"
)
