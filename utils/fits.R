# libraries and utils
library(tidyverse)

source("./utils/bayes.R")
source("./utils/matching.R")

# data -------------------------------------------------------------------------------
load_metrics <- function(band) {
  read_csv(paste0("./data/connectome_metrics_", band, ".csv"), show_col_types = FALSE)
}

# Model 1: per-child differences --------------------------------------------------------
fit_model1 <- function(model, df, metric, analysis, band, covariates = character(0), mu_sd = 1) {
  pcd <- per_child_differences(df, metric, covariates)
  fit <- sample_model(model, model1_data(pcd, covariates, mu_sd))

  pooled_sd <- pcd$differences$pooled_sd[1]
  std <- summarise_param(fit, "mu")
  raw_scale <- summarise_param(fit, "mu", scale = pooled_sd)

  list(
    summary = tibble(
      analysis = analysis,
      band = band,
      metric = metric,
      n_test = nrow(pcd$differences),
      n_control = pcd$n_controls,
      n_strata = nrow(pcd$strata),
      pooled_sd = pooled_sd,
      mu = std$mean,
      mu_lower = std$lower,
      mu_upper = std$upper,
      raw = raw_scale$mean,
      raw_lower = raw_scale$lower,
      raw_upper = raw_scale$upper,
      p_below = std$p_below,
      p_below_mcse = std$p_below_mcse,
      n_above = std$n_above,
      n_below = std$n_below,
      ess_tail = std$ess_tail,
      problems = std$problems,
      refit = std$refit
    ),
    differences = pcd$differences %>%
      mutate(analysis = analysis, band = band, metric = metric),
    fit = fit
  )
}

# Model 2: group x age ------------------------------------------------------------------
model2_data <- function(df, metric) {
  controls <- df %>% filter(group == "control")
  y_mean <- mean(controls[[metric]])
  y_sd <- sd(controls[[metric]])
  age_sd <- sd(df$age)

  list(
    data = list(
      n = nrow(df),
      y = (df[[metric]] - y_mean) / y_sd,
      group = as.numeric(df$group == "test"),
      age = (df$age - 5) / (2 * age_sd),
      sex = ifelse(df$sex == "M", 0.5, -0.5)
    ),
    y_mean = y_mean,
    y_sd = y_sd,
    age_sd = age_sd
  )
}

# posterior draws of Model 2 ------------------------------------------------------------
model2_draws <- function(fit) {
  draws <- as_draws_df(fit$draws(c("alpha", "b_group", "b_age", "b_group_age")))
  tibble(
    alpha = draws$alpha,
    b_group = draws$b_group,
    b_age = draws$b_age,
    b_group_age = draws$b_group_age
  )
}

fit_model2 <- function(model, df, metric, analysis, band) {
  prep <- model2_data(df, metric)
  fit <- sample_model(model, prep$data)

  # slopes per year, in control SD units
  per_year <- 1 / (2 * prep$age_sd)

  summarise_draws_vec <- function(x, name) {
    below <- as.numeric(x < 0)
    tibble(
      parameter = name,
      mean = mean(x),
      lower = unname(quantile(x, 0.025)),
      upper = unname(quantile(x, 0.975)),
      p_below = mean(below),
      p_below_mcse = mcse_mean(below),
      n_above = sum(x >= 0),
      n_below = sum(x < 0)
    )
  }

  draws <- model2_draws(fit)
  out <- bind_rows(
    summarise_draws_vec(draws$b_group, "group (age 5)"),
    summarise_draws_vec(draws$b_age * per_year, "slope control (per year)"),
    summarise_draws_vec((draws$b_age + draws$b_group_age) * per_year, "slope test (per year)"),
    summarise_draws_vec(draws$b_group_age * per_year, "group x age (per year)")
  )

  summary <- out %>%
    mutate(
      analysis = analysis,
      band = band,
      metric = metric,
      n_test = sum(df$group == "test"),
      n_control = sum(df$group == "control"),
      y_sd = prep$y_sd,
      problems = attr(fit, "problems"),
      refit = attr(fit, "refit"),
      .before = 1
    )

  list(summary = summary, fit = fit, prep = prep)
}
