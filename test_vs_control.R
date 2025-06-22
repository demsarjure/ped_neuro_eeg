# libraries and utils ----------------------------------------------------------
library(tidyverse)

source("./utils/normal.R")

# load data --------------------------------------------------------------------
band <- "alpha"
df_metrics <- read_csv(paste0("./data/connectome_metrics_", band, ".csv"))

# pairs ------------------------------------------------------------------------
age_difference <- 0

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
    lh_rh = lh_rh_test - lh_rh_control,
    laq_raq = laq_raq_test - laq_raq_control,
    lpq_rpq = lpq_rpq_test - lpq_rpq_control,
    laq_rpq = laq_rpq_test - laq_rpq_control,
    raq_lpq = raq_lpq_test - raq_lpq_control,
    ge = ge_test - ge_control,
    age_difference = abs(age_test - age_control)
  )

# subset df_test and df_control to only include ids in df_pairs
paste0("Test size before: ", nrow(df_test))
paste0("Control size before: ", nrow(df_control))

# demographics
paste0("Min age test: ", min(df_test$age))
paste0("Min age control: ", min(df_control$age))
paste0("Max age test: ", max(df_test$age))
paste0("Max age control: ", max(df_control$age))
paste0(
  "Test: M =  ", nrow(df_test[df_test$sex == "M", ]),
  ", F = ", nrow(df_test[df_test$sex != "M", ])
)
paste0(
  "Control: M =  ", nrow(df_control[df_control$sex == "M", ]),
  ", F = ", nrow(df_control[df_control$sex != "M", ])
)

# lh_rh ------------------------------------------------------------------------
fit_lh_rh <- fit_normal(df_pairs$lh_rh)
results <- compare_normal(
  fit = fit_lh_rh, label1 = "test", label2 = "control"
)

# laq_raq ----------------------------------------------------------------------
fit_laq_raq <- fit_normal(df_pairs$laq_raq)
results <- compare_normal(
  fit = fit_laq_raq, label1 = "test", label2 = "control"
)

# lpq_rpq ----------------------------------------------------------------------
fit_lpq_rpq <- fit_normal(df_pairs$lpq_rpq)
results <- compare_normal(
  fit = fit_lpq_rpq, label1 = "test", label2 = "control"
)

# laq_rpq ----------------------------------------------------------------------
fit_laq_rpq <- fit_normal(df_pairs$laq_rpq)
results <- compare_normal(
  fit = fit_laq_rpq, label1 = "test", label2 = "control"
)

# raq_lpq ----------------------------------------------------------------------
fit_raq_lpq <- fit_normal(df_pairs$raq_lpq)
results <- compare_normal(
  fit = fit_raq_lpq, label1 = "test", label2 = "control"
)

# ge ---------------------------------------------------------------------------
fit_ge <- fit_normal(df_pairs$ge)
results <- compare_normal(
  fit = fit_ge, label1 = "test", label2 = "control"
)
plot_comparison_normal(fit = fit_ge) +
  ggtitle("Global efficiency") +
  xlim(-0.005, 0.005) +
  xlab("Mean difference") +
  theme(plot.title = element_text(hjust = 0.5))
ggsave(
  paste0("./figures/ge_", band, ".png"),
  width = 1920,
  height = 1080,
  dpi = 150,
  units = "px",
  bg = "white"
)
