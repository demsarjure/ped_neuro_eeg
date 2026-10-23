# libraries and utils ----------------------------------------------------------
library(tidyverse)

source("./utils/fits.R")

dir.create("./results", showWarnings = FALSE)

# robustness of the primary result: the checks and the minimum posterior
# probability of the primary's sign in each of them
ROBUST_ANALYSES <- c(
  "S1 individual band", "S2 age 0-4", "S2 age 5+", "S3 without slowing",
  "S4 without sleep", "S11 without frontal electrodes"
)
ROBUST_P_MIN <- 0.90

model1 <- compile_model("diff_t")
model2 <- compile_model("group_age_t")
matched_set <- compile_model("matched_set_t")

alpha <- load_metrics("alpha")
pdr <- read_csv("./data/pdr.csv", show_col_types = FALSE)
qc <- read_csv("./data/qc_summary.csv", show_col_types = FALSE)

# subsets filter both groups before matching, so flagged controls also leave the
# control means
results <- list()
add <- function(r) results[[length(results) + 1]] <<- r$summary

# S2: age strata -------------------------------------------------------------------
add(fit_model1(model1, alpha %>% filter(age <= 4), "lh_rh", "S2 age 0-4", "alpha"))
add(fit_model1(model1, alpha %>% filter(age >= 5), "lh_rh", "S2 age 5+", "alpha"))

# S3: without children flagged by the background-slowing proxy (both groups) -------
slowed <- pdr %>% filter(slowing_flag) %>% pull(id)
add(fit_model1(model1, alpha %>% filter(!id %in% slowed), "lh_rh", "S3 without slowing", "alpha"))

# S4: without recordings with signs of sleep (spindles and no clear PDR) ------------
asleep <- pdr %>% filter(sleep_flag) %>% pull(id)
add(fit_model1(model1, alpha %>% filter(!id %in% asleep), "lh_rh", "S4 without sleep", "alpha"))

# S7: global efficiency is fitted in analysis_group.R (exploratory metrics)

# S8a: data-quality covariates (muscle activity and artefact removal); S8b: also
# alpha power, which may remove part of a genuine effect (mediation) -----------------
alpha_cov <- alpha %>%
  left_join(qc %>% select(id, hf_power_db, n_ica_excluded), by = "id") %>%
  left_join(pdr %>% select(id, alpha_adj_db), by = "id")
add(fit_model1(
  model1, alpha_cov, "lh_rh", "S8a data quality", "alpha",
  covariates = c("hf_power_db", "n_ica_excluded")
))
add(fit_model1(
  model1, alpha_cov, "lh_rh", "S8b data quality and alpha power", "alpha",
  covariates = c("hf_power_db", "n_ica_excluded", "alpha_adj_db")
))

# S9: without recordings in which Autoreject's thresholds fell below the channels'
# median amplitude (mis-calibration, heavy interpolation) ----------------------------
miscalibrated <- qc %>% filter(!excluded, ar_threshold_ratio < 1) %>% pull(id)
add(fit_model1(model1, alpha %>% filter(!id %in% miscalibrated), "lh_rh", "S9 without Autoreject mis-calibration", "alpha"))

# S10: without electrode pairs involving an interpolated electrode ------------------
add(fit_model1(model1, alpha, "lh_rh_noint", "S10 without interpolated electrodes", "alpha"))

# S11: without the frontal-pole and anterior temporal electrodes (Fp1, Fp2, F7, F8)
add(fit_model1(model1, alpha, "lh_rh_nofront", "S11 without frontal electrodes", "alpha"))

# prior sensitivity ----------------------------------------------------------------
add(fit_model1(model1, alpha, "lh_rh", "prior mu ~ N(0, 0.5)", "alpha", mu_sd = 0.5))
add(fit_model1(model1, alpha, "lh_rh", "prior mu ~ N(0, 2)", "alpha", mu_sd = 2))

model1_results <- bind_rows(results)

# S5: all analysable children, spline of age -----------------------------------------
s5 <- fit_model2(model2, alpha %>% filter(age <= 19), "lh_rh", "S5 all children", "alpha", spline_df = 3)$summary

# check: matched-set model -----------------------------------------------------------
# the outcome is centred on the matched controls' mean, so that the prior on the
# stratum levels does not pull mu toward 0
strata <- matched_strata(alpha)
pcd <- per_child_differences(alpha, "lh_rh")
pooled_sd <- pcd$differences$pooled_sd[1]
ms <- alpha %>%
  inner_join(strata, by = c("sex", "age")) %>%
  mutate(y = (lh_rh - mean(lh_rh[group == "control"])) / pooled_sd)
ms_fit <- sample_model(
  matched_set,
  list(
    n_control = sum(ms$group == "control"),
    n_test = sum(ms$group == "test"),
    n_strata = nrow(strata),
    stratum_control = ms$stratum[ms$group == "control"],
    stratum_test = ms$stratum[ms$group == "test"],
    y_control = ms$y[ms$group == "control"],
    y_test = ms$y[ms$group == "test"]
  )
)
ms_summary <- summarise_param(ms_fit, "mu") %>%
  mutate(analysis = "check: matched-set model", band = "alpha", metric = "lh_rh", .before = 1)

# robustness of the primary result (S1 and the primary are in model1.csv) ------------
group_results <- read_csv("./results/model1.csv", show_col_types = FALSE)
primary <- group_results %>% filter(analysis == "primary")
robustness <- bind_rows(group_results, model1_results) %>%
  filter(analysis %in% ROBUST_ANALYSES) %>%
  transmute(
    analysis, n_test, n_control, mu, mu_lower, mu_upper,
    p_same_sign = if (primary$mu < 0) p_below else 1 - p_below
  )
stopifnot(nrow(robustness) == length(ROBUST_ANALYSES))
robustness$label <- robust_label(primary, robustness$p_same_sign, ROBUST_P_MIN)

# save -------------------------------------------------------------------------------
write_csv(model1_results, "./results/sensitivity_model1.csv")
write_csv(s5, "./results/sensitivity_s5.csv")
write_csv(ms_summary, "./results/check_matched_set.csv")
write_csv(robustness, "./results/robustness.csv")

print(model1_results %>% select(analysis, n_test, n_control, mu, mu_lower, mu_upper, p_below, problems))
print(s5 %>% select(analysis, parameter, mean, lower, upper, p_below))
print(ms_summary %>% select(analysis, mean, lower, upper, p_below))
print(robustness)
print(paste("Primary result:", robustness$label[1]))
