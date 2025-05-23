library(tidyverse)

# load data and models ---------------------------------------------------------
source("./utils/normal.R")

df_metrics <- read_csv("./data/connectome_metrics.csv")


# pairs ------------------------------------------------------------------------
age_difference <- 1

df_test <- df_metrics %>%
  filter(group == "test")

df_control <- df_metrics %>%
  filter(group == "control")

df_pairs <- df_test %>%
  inner_join(df_control, by = "sex", suffix = c("_test", "_control")) %>%
  filter(abs(age_test - age_control) <= age_difference) %>%
  transmute(
    id = id_test,
    id_control = id_control,
    ihs = ihs_test - ihs_control,
    ge = ge_test - ge_control,
    age_difference= abs(age_test - age_control)
  )

# ihs --------------------------------------------------------------------------
fit_ihs <- fit_normal(df_pairs$ihs)
results <- compare_normal(
    fit = fit_ihs, label1 = "test", label2 = "control"
)
plot_comparison_normal(fit = fit_ihs)

# ge --------------------------------------------------------------------------
fit_ge <- fit_normal(df_pairs$ge)
results <- compare_normal(
    fit = fit_ge, label1 = "test", label2 = "control"
)
plot_comparison_normal(fit = fit_ge)
