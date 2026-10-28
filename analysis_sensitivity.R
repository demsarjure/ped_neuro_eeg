# libraries and utils ----------------------------------------------------------
library(tidyverse)

source("./utils/fits.R")

dir.create("./results", showWarnings = FALSE)

model1 <- compile_model("diff_t")

alpha <- load_metrics("alpha")
pdr <- read_csv("./data/pdr.csv", show_col_types = FALSE)
qc <- read_csv("./data/qc_summary.csv", show_col_types = FALSE)

# subsets filter both groups before matching, so flagged controls also leave the
# control means
results <- list()
add <- function(r) results[[length(results) + 1]] <<- r$summary

# S2: age strata -------------------------------------------------------------------
add(fit_model1(model1, alpha %>% filter(age <= 4), "lh_rh", "S2 age 2-4", "alpha"))
add(fit_model1(model1, alpha %>% filter(age >= 5), "lh_rh", "S2 age 5+", "alpha"))

# S3: without children flagged by the background-slowing proxy (both groups) -------
slowed <- pdr %>% filter(slowing_flag) %>% pull(id)
add(fit_model1(model1, alpha %>% filter(!id %in% slowed), "lh_rh", "S3 without slowing", "alpha"))

# S8a: data-quality covariates (muscle activity and artefact removal) ---------------
alpha_cov <- alpha %>%
  left_join(qc %>% select(id, hf_power_db, n_ica_excluded), by = "id")
add(fit_model1(
  model1, alpha_cov, "lh_rh", "S8a data quality", "alpha",
  covariates = c("hf_power_db", "n_ica_excluded")
))

# S11: without the frontal-pole and anterior temporal electrodes (Fp1, Fp2, F7, F8)
add(fit_model1(model1, alpha, "lh_rh_nofront", "S11 without frontal electrodes", "alpha"))

# save -------------------------------------------------------------------------------
model1_results <- bind_rows(results)
write_csv(model1_results, "./results/sensitivity_model1.csv")

print(model1_results %>% select(analysis, n_test, n_control, mu, mu_lower, mu_upper, p_below, problems))
