# libraries and utils ----------------------------------------------------------
library(tidyverse)

source("./utils/simple_linear.R")

# load data --------------------------------------------------------------------
df_metrics <- read_csv("./data/connectome_metrics.csv")
df_metrics_test <- df_metrics %>%
  filter(group == "test")
df_metrics_control <- df_metrics %>%
  filter(group == "control")

# ge ---------------------------------------------------------------------------
# metric
y_test_ge <- df_metrics_test$ge
y_control_ge <- df_metrics_control$ge

# test
df_test_ge <- data.frame(
  x = df_metrics_test$age,
  y = y_test_ge
)
fit_test <- fit_simple_linear(
  x = df_test_ge$x,
  y = df_test_ge$y
)
compare_simple_linear(fit = fit_test)
plot_simple_linear(fit = fit_test, data = df_test_ge)

# control
df_control_ge <- data.frame(
  x = df_metrics_control$age,
  y = y_control_ge
)
fit_control <- fit_simple_linear(
  x = df_control_ge$x,
  y = df_control_ge$y
)
compare_simple_linear(fit = fit_control)
plot_simple_linear(fit = fit_control, data = df_control_ge)

# lh_rh ------------------------------------------------------------------------
# metric
y_test_lh_rh <- df_metrics_test$lh_rh
y_control_lh_rh <- df_metrics_control$lh_rh

# test
df_test_lh_rh <- data.frame(
  x = df_metrics_test$age,
  y = y_test_lh_rh
)
fit_test <- fit_simple_linear(
  x = df_test_lh_rh$x,
  y = df_test_lh_rh$y
)
compare_simple_linear(fit = fit_test)
plot_simple_linear(fit = fit_test, data = df_test_lh_rh)

# control
df_control_lh_rh <- data.frame(
  x = df_metrics_control$age,
  y = y_control_lh_rh
)
fit_control <- fit_simple_linear(
  x = df_control_lh_rh$x,
  y = df_control_lh_rh$y
)
compare_simple_linear(fit = fit_control)
plot_simple_linear(fit = fit_control, data = df_control_lh_rh)
