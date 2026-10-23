# libraries and utils ----------------------------------------------------------
library(tidyverse)

source("./utils/fits.R")

dir.create("./results", showWarnings = FALSE)

# Model 2 sample: the ages covered by both groups
AGE_MIN <- 2
AGE_MAX <- 9

model2 <- compile_model("group_age_t")

alpha <- load_metrics("alpha")
young <- alpha %>% filter(age >= AGE_MIN, age <= AGE_MAX)

# matched sample only (check)
strata <- matched_strata(young)
matched <- young %>% semi_join(strata, by = c("sex", "age"))

# key secondary: IHS, all analysable children aged 2-9
ihs <- fit_model2(model2, young, "lh_rh", "key secondary", "alpha")
# check: matched children only
ihs_matched <- fit_model2(model2, matched, "lh_rh", "check: matched only", "alpha")
# exploratory: global efficiency
ge <- fit_model2(model2, young, "ge", "exploratory", "alpha")

results <- bind_rows(ihs$summary, ihs_matched$summary, ge$summary)
write_csv(results, "./results/model2.csv")

# regression lines for the figures
lines <- bind_rows(
  model2_lines(ihs$fit, ihs$prep, young) %>% mutate(metric = "lh_rh"),
  model2_lines(ge$fit, ge$prep, young) %>% mutate(metric = "ge")
)
write_csv(lines, "./results/model2_lines.csv")

# influence: the key secondary refitted without each CTNNB1 child in turn ---------
# a child is influential if leaving it out changes the sign of beta_ga or whether
# its 95% credible interval excludes 0
excludes_zero <- function(lower, upper) lower > 0 | upper < 0
full <- ihs$summary %>% filter(parameter == "group x age (per year)")

influence <- map_dfr(young$id[young$group == "test"], function(left_out) {
  fit_model2(model2, young %>% filter(id != left_out), "lh_rh", "influence", "alpha")$summary %>%
    filter(parameter == "group x age (per year)") %>%
    mutate(left_out = left_out, .before = 1)
}) %>%
  mutate(
    sign_changed = sign(mean) != sign(full$mean),
    cri_changed = excludes_zero(lower, upper) != excludes_zero(full$lower, full$upper),
    influential = sign_changed | cri_changed
  )
write_csv(influence, "./results/model2_influence.csv")

print(results %>% select(analysis, metric, parameter, mean, lower, upper, p_below, problems))
influential_ids <- influence %>% filter(influential) %>% pull(left_out)
print(paste(
  "Influential CTNNB1 children:",
  if (length(influential_ids) > 0) paste(influential_ids, collapse = ", ") else "none"
))
