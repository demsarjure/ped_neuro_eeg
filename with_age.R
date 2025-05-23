library(tidyverse)

# load data and models ---------------------------------------------------------
source("./utils/normal.R")

df_metrics <- read_csv("./data/connectome_metrics.csv")
df_metrics_test <- df_metrics %>%
  filter(group == "test")
df_metrics_control <- df_metrics %>%
  filter(group == "control")

# test -------------------------------------------------------------------------
# metric
y_test <- df_metrics_test$ihs
y_control <- df_metrics_control$ihs

# test
df_test <- data.frame(
  x = df_metrics_test$age,
  y = y_test
)
fit_test <- fit_simple_linear(
  x = df_test$x,
  y = df_test$y
)
compare_simple_linear(fit = fit_test)
plot_simple_linear(fit = fit_test, data = df_test)

# control
df_control <- data.frame(
  x = df_metrics_control$age,
  y = y_control
)
fit_control <- fit_simple_linear(
  x = df_control$x,
  y = df_control$y
)
compare_simple_linear(fit = fit_control)
plot_simple_linear(fit = fit_control, data = df_control)
