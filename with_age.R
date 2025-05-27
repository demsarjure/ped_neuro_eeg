# libraries and utils ----------------------------------------------------------
library(tidyverse)
library(cowplot)

source("./utils/simple_linear.R")

# load data --------------------------------------------------------------------
df_metrics <- read_csv("./data/connectome_metrics.csv")
df_metrics_test <- df_metrics %>%
  filter(group == "test")
df_metrics_control <- df_metrics %>%
  filter(group == "control")

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
  ylab("Global efficiency (GE)")

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
  ylab("Global efficiency (GE)")

plot_grid(plotlist = ge_plots, ncol = 2, scale = 0.9)
ggsave(
  "./figures/ge_age.png",
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
  "./figures/ihs_age.png",
  width = 1920,
  height = 1080,
  dpi = 150,
  units = "px",
  bg = "white"
)
