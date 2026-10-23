# libraries
library(tidyverse)

# matched strata -------------------------------------------------------------------
# every test child is matched to all controls of the same sex and the same age
# in completed years; test children without a matching control are dropped
matched_strata <- function(df) {
  controls <- df %>%
    filter(group == "control") %>%
    count(sex, age, name = "m")

  df %>%
    filter(group == "test") %>%
    inner_join(controls, by = c("sex", "age")) %>%
    distinct(sex, age, m) %>%
    arrange(age, sex) %>%
    mutate(stratum = row_number())
}

# per-child differences --------------------------------------------------------------
# test child minus the mean of its matched controls, standardised by the pooled
# within-stratum SD of the matched controls (one value per metric)
per_child_differences <- function(df, metric, covariates = character(0)) {
  strata <- matched_strata(df)

  controls <- df %>%
    filter(group == "control") %>%
    inner_join(strata, by = c("sex", "age"))
  tests <- df %>%
    filter(group == "test") %>%
    inner_join(strata, by = c("sex", "age"))

  standardise <- function(variable) {
    pooled_sd <- controls %>%
      group_by(stratum) %>%
      filter(n() > 1) %>%
      summarise(ss = sum((.data[[variable]] - mean(.data[[variable]]))^2), dfree = n() - 1) %>%
      summarise(sd = sqrt(sum(ss) / sum(dfree))) %>%
      pull(sd)

    control_means <- controls %>%
      group_by(stratum) %>%
      summarise(control_mean = mean(.data[[variable]]))

    tests %>%
      select(id, stratum, value = all_of(variable)) %>%
      left_join(control_means, by = "stratum") %>%
      transmute(id, stratum, raw = value - control_mean, d = raw / pooled_sd, pooled_sd = pooled_sd)
  }

  out <- standardise(metric) %>%
    left_join(tests %>% select(id, sex, age, m), by = "id")

  for (covariate in covariates) {
    cov_d <- standardise(covariate) %>% select(id, !!covariate := d)
    out <- out %>% left_join(cov_d, by = "id")
  }

  list(
    differences = out,
    strata = strata,
    n_controls = n_distinct(controls$id)
  )
}

# data for Model 1 ----------------------------------------------------------------------
model1_data <- function(pcd, covariates = character(0), mu_sd = 1) {
  x <- as.matrix(pcd$differences[, covariates, drop = FALSE])

  list(
    n = nrow(pcd$differences),
    n_strata = nrow(pcd$strata),
    stratum = pcd$differences$stratum,
    m = pcd$strata$m,
    d = pcd$differences$d,
    k = length(covariates),
    x = x,
    mu_sd = mu_sd
  )
}
