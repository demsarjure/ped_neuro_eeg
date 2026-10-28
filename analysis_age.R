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

# key secondary: IHS, all analysable children aged 2-9
ihs <- fit_model2(model2, young, "lh_rh", "key secondary", "alpha")
# exploratory: global efficiency
ge <- fit_model2(model2, young, "ge", "exploratory", "alpha")

results <- bind_rows(ihs$summary, ge$summary)
write_csv(results, "./results/model2.csv")

print(results %>% select(analysis, metric, parameter, mean, lower, upper, p_below, problems))
