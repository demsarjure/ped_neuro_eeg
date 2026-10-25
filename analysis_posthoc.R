# Post hoc analyses, not part of the analysis plan (25 October 2026) --------------------
# A1: alpha power. Children whose posterior alpha power lies in the range observed in
#     both groups, compared with the alpha-power difference as a covariate, so that
#     the adjustment does not extrapolate beyond the observed differences (unlike S8b).
# A2: connectivity within the hemispheres, and the within-child contrast between
#     interhemispheric and intrahemispheric strength (the same 16 electrodes).
# A3: eye state. Children with a blink within 15 s of at least 90% of their analysed
#     epochs (blinks_posthoc.py), whose eyes were open in nearly all of them.
# A4: signal-to-noise. The matched control children with noise added until their
#     alpha power matched that of the CTNNB1 children of their stratum (noise_posthoc.py).
# A5: the key secondary age model (Model 2) in the individual band.
# A6: influence of single control children and strata on the main estimate, the
#     placement of each CTNNB1 child within its matched control children, the
#     strata, and posterior alpha power as an outcome.
# A7: without the two control children furthest above their stratum mean, without the
#     control children with a spindle rate of at least 0.5/min, a probabilistic index
#     with a Bayesian-bootstrap interval, and checks of the age model (individual band
#     in children with a clear rhythm; leaving out one control child at a time).

# libraries and utils ----------------------------------------------------------
library(tidyverse)

source("./utils/fits.R")

dir.create("./results", showWarnings = FALSE)

model1 <- compile_model("diff_t")
model2 <- compile_model("group_age_t")

pdr <- read_csv("./data/pdr.csv", show_col_types = FALSE)
alpha <- load_metrics("alpha") %>%
  mutate(lh_rh_minus_intra = lh_rh - intra) %>%
  left_join(pdr %>% select(id, alpha_adj_db), by = "id")

results <- list()
add <- function(r) results[[length(results) + 1]] <<- r$summary

# A2: within the hemispheres -----------------------------------------------------------
add(fit_model1(model1, alpha, "intra", "post hoc: within the hemispheres", "alpha"))
add(fit_model1(model1, alpha, "lh_rh_minus_intra", "post hoc: between minus within the hemispheres", "alpha"))

# A1: overlapping alpha power, adjusted for it ------------------------------------------
# both groups are restricted to the range of posterior alpha power observed in both
# before matching, as in S3
alpha_range <- alpha %>%
  group_by(group) %>%
  summarise(low = min(alpha_adj_db), high = max(alpha_adj_db)) %>%
  summarise(low = max(low), high = min(high))
in_range <- alpha %>%
  filter(alpha_adj_db >= alpha_range$low, alpha_adj_db <= alpha_range$high)
overlap_fit <- fit_model1(
  model1, in_range, "lh_rh", "post hoc: overlapping alpha power, adjusted", "alpha",
  covariates = c("alpha_adj_db")
)
add(overlap_fit)

# A3: blinks near nearly all analysed epochs ---------------------------------------------
BLINK_MIN_FRACTION <- 0.9
blinks <- read_csv("./data/blinks_analysed.csv", show_col_types = FALSE)
eyes_open <- blinks %>% filter(epochs_near_blink >= BLINK_MIN_FRACTION) %>% pull(id)
add(fit_model1(model1, alpha %>% filter(id %in% eyes_open), "lh_rh", "post hoc: blinks near the analysed epochs", "alpha"))
# other thresholds and a stricter window, as checks
for (rule in list(c("epochs_near_blink", 0.5), c("epochs_near_blink", 0.75), c("epochs_near_blink_5s", 0.5))) {
  ids <- blinks %>% filter(.data[[rule[1]]] >= as.numeric(rule[2])) %>% pull(id)
  add(fit_model1(
    model1, alpha %>% filter(id %in% ids), "lh_rh",
    paste0("post hoc: blinks check (", rule[1], " >= ", rule[2], ")"), "alpha"
  ))
}

# A4: matched control children degraded to the alpha power of their stratum -------------
noise <- read_csv("./data/noise_injection.csv", show_col_types = FALSE)
alpha_noise <- alpha %>%
  left_join(noise %>% select(id, lh_rh_noise), by = "id") %>%
  mutate(lh_rh = if_else(group == "control" & !is.na(lh_rh_noise), lh_rh_noise, lh_rh))
add(fit_model1(model1, alpha_noise, "lh_rh", "post hoc: control alpha power degraded with noise", "alpha"))

# the adjustment estimates the difference at an alpha-power difference of zero, which
# should lie within the observed differences
alpha_diff <- overlap_fit$differences$alpha_adj_db

# A8: epochs matched for alpha power (epochs_posthoc.py): IHS from the 50 epochs with the
#     least (control children) or most (CTNNB1 children) posterior alpha power ----------
epochs_matched <- read_csv("./data/epochs_alpha_matched.csv", show_col_types = FALSE)
add(fit_model1(
  model1, alpha %>% left_join(epochs_matched %>% select(id, lh_rh_alpha_matched_epochs), by = "id"),
  "lh_rh_alpha_matched_epochs", "post hoc: epochs matched for alpha power", "alpha"
))

# A6: posterior alpha power as an outcome ------------------------------------------------
add(fit_model1(model1, alpha, "alpha_adj_db", "post hoc: posterior alpha power", "alpha"))

# A6: influence of single control children and strata on the main estimate -------------
strata <- matched_strata(alpha)
matched_controls <- alpha %>%
  filter(group == "control") %>%
  semi_join(strata, by = c("sex", "age")) %>%
  pull(id)
loo_control <- map_dfr(matched_controls, function(left_out) {
  fit_model1(model1, alpha %>% filter(id != left_out), "lh_rh", "leave one control out", "alpha")$summary %>%
    mutate(left_out = left_out, .before = 1)
})
loo_stratum <- map_dfr(seq_len(nrow(strata)), function(s) {
  fit_model1(
    model1, alpha %>% anti_join(strata[s, ], by = c("sex", "age")), "lh_rh", "leave one stratum out", "alpha"
  )$summary %>%
    mutate(left_out = paste(strata$sex[s], strata$age[s]), .before = 1)
})

# placement: each CTNNB1 child's percentile among its matched control children (ties
# count half), and the strata
placement <- alpha %>%
  filter(group == "test") %>%
  inner_join(strata, by = c("sex", "age")) %>%
  rowwise() %>%
  mutate(percentile = {
    controls <- alpha$lh_rh[alpha$group == "control" & alpha$sex == sex & alpha$age == age]
    mean((controls < lh_rh) + 0.5 * (controls == lh_rh))
  }) %>%
  ungroup() %>%
  select(id, sex, age, percentile)
strata_table <- strata %>%
  left_join(alpha %>% filter(group == "test") %>% count(sex, age, name = "n_test"), by = c("sex", "age")) %>%
  left_join(
    alpha %>%
      filter(group == "control") %>%
      group_by(sex, age) %>%
      summarise(n_control = n(), control_mean = mean(lh_rh), control_median = median(lh_rh), .groups = "drop"),
    by = c("sex", "age")
  ) %>%
  select(sex, age, n_test, n_control, control_mean, control_median)

# A7: the two control children furthest above their stratum mean, removed together ------
pcd <- per_child_differences(alpha, "lh_rh")
deviations <- alpha %>%
  filter(group == "control") %>%
  inner_join(strata, by = c("sex", "age")) %>%
  group_by(stratum) %>%
  mutate(z = (lh_rh - mean(lh_rh)) / pcd$differences$pooled_sd[1]) %>%
  ungroup() %>%
  arrange(desc(z))
top_two <- head(deviations$id, 2)
add(fit_model1(model1, alpha %>% filter(!id %in% top_two), "lh_rh", "post hoc: without the two highest control children", "alpha"))

# A7: without control children with spindles (rate of at least 0.5/min) ----------------
qc <- read_csv("./data/qc_summary.csv", show_col_types = FALSE)
spindles <- qc %>% filter(spindle_flag) %>% pull(id)
add(fit_model1(model1, alpha %>% filter(!id %in% spindles), "lh_rh", "post hoc: without spindles", "alpha"))

# A7: probabilistic index: the probability that a CTNNB1 child's IHS is below that of a
# matched control child (ties count half), averaged over the matched pairs, with a
# Bayesian bootstrap over the children of both groups
set.seed(42)
pairs <- alpha %>%
  filter(group == "test") %>%
  inner_join(strata, by = c("sex", "age")) %>%
  select(test_id = id, sex, age, y_test = lh_rh) %>%
  inner_join(alpha %>% filter(group == "control") %>% select(control_id = id, sex, age, y_control = lh_rh), by = c("sex", "age")) %>%
  mutate(below = (y_test < y_control) + 0.5 * (y_test == y_control))
test_ids <- unique(pairs$test_id)
control_ids <- unique(pairs$control_id)
pi_draws <- replicate(10000, {
  w_test <- setNames(rgamma(length(test_ids), 1), test_ids)
  w_control <- setNames(rgamma(length(control_ids), 1), control_ids)
  w <- w_test[pairs$test_id] * w_control[pairs$control_id]
  sum(w * pairs$below) / sum(w)
})
prob_index <- tibble(
  analysis = "post hoc: probabilistic index",
  estimate = mean(pairs$below),
  lower = unname(quantile(pi_draws, 0.025)),
  upper = unname(quantile(pi_draws, 0.975)),
  n_pairs = nrow(pairs)
)

# A5: age model in the individual band ---------------------------------------------------
age_iaf <- bind_rows(
  fit_model2(model2, load_metrics("iaf"), "lh_rh", "post hoc: key secondary in the individual band", "iaf")$summary,
  # the band follows the child's own rhythm only where one was found
  fit_model2(
    model2, load_metrics("iaf") %>% semi_join(pdr %>% filter(pdr_clear), by = "id"), "lh_rh",
    "post hoc: key secondary in the individual band, clear rhythm only", "iaf"
  )$summary
)

# A7: the key secondary refitted without each control child in turn ---------------------
age_loo_control <- map_dfr(alpha$id[alpha$group == "control"], function(left_out) {
  fit_model2(model2, alpha %>% filter(id != left_out), "lh_rh", "influence of control children", "alpha")$summary %>%
    filter(parameter == "group x age (per year)") %>%
    mutate(left_out = left_out, .before = 1)
})

posthoc <- bind_rows(results) %>%
  mutate(
    alpha_low = alpha_range$low,
    alpha_high = alpha_range$high,
    alpha_diff_min = if_else(analysis == "post hoc: overlapping alpha power, adjusted", min(alpha_diff), NA_real_),
    alpha_diff_max = if_else(analysis == "post hoc: overlapping alpha power, adjusted", max(alpha_diff), NA_real_)
  )

# save ---------------------------------------------------------------------------------
write_csv(posthoc, "./results/posthoc.csv")
write_csv(age_iaf, "./results/posthoc_age.csv")
write_csv(bind_rows(loo_control, loo_stratum), "./results/posthoc_influence.csv")
write_csv(placement, "./results/posthoc_placement.csv")
write_csv(strata_table, "./results/strata.csv")
write_csv(prob_index, "./results/posthoc_probability_index.csv")
write_csv(age_loo_control, "./results/posthoc_age_influence.csv")

print(posthoc %>% select(analysis, n_test, n_control, mu, mu_lower, mu_upper, p_below, problems))
print(age_iaf %>% select(analysis, parameter, mean, lower, upper, p_below, problems))
print(paste("Alpha power range in both groups:", round(alpha_range$low, 2), "to", round(alpha_range$high, 2), "dB"))
print(paste("Alpha-power differences (control SDs):", round(min(alpha_diff), 2), "to", round(max(alpha_diff), 2)))
