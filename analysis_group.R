# libraries and utils ----------------------------------------------------------
library(tidyverse)

source("./utils/fits.R")

dir.create("./results", showWarnings = FALSE)

model1 <- compile_model("diff_t")

# primary and exploratory metrics (8-13 Hz) --------------------------------------
alpha <- load_metrics("alpha")
metrics <- c("lh_rh", "laq_raq", "lpq_rpq", "laq_rpq", "raq_lpq", "ge", "mean_conn")

results <- list()
for (metric in metrics) {
  analysis <- ifelse(metric == "lh_rh", "primary", "exploratory")
  results[[length(results) + 1]] <- fit_model1(model1, alpha, metric, analysis, "alpha")
}

# S1: individual band ------------------------------------------------------------
results[[length(results) + 1]] <- fit_model1(
  model1, load_metrics("iaf"), "lh_rh", "S1 individual band", "iaf"
)

# S6: theta ----------------------------------------------------------------------
results[[length(results) + 1]] <- fit_model1(
  model1, load_metrics("theta"), "lh_rh", "S6 theta", "theta"
)

# save -----------------------------------------------------------------------------
summary <- bind_rows(map(results, "summary"))
differences <- bind_rows(map(results, "differences"))
write_csv(summary, "./results/model1.csv")
write_csv(differences, "./results/model1_differences.csv")

# posterior predictive check of the primary model
primary <- results[[1]]
d <- primary$differences$d
d_rep <- as_draws_matrix(primary$fit$draws("d_rep"))
ppc <- tibble(
  statistic = c("mean", "sd", "min", "max"),
  observed = c(mean(d), sd(d), min(d), max(d)),
  p_rep_greater = c(
    mean(apply(d_rep, 1, mean) > mean(d)),
    mean(apply(d_rep, 1, sd) > sd(d)),
    mean(apply(d_rep, 1, min) > min(d)),
    mean(apply(d_rep, 1, max) > max(d))
  )
)
write_csv(ppc, "./results/model1_ppc.csv")

print(summary %>% select(analysis, band, metric, n_test, n_control, mu, mu_lower, mu_upper, p_below, problems))
