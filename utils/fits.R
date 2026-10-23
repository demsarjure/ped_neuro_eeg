# libraries and utils
library(tidyverse)
library(splines)

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

# robustness of the primary result ---------------------------------------------------
# assessed only if the primary 95% credible interval excludes 0; robust if the
# posterior probability of the primary's sign is at least p_min in every check
robust_label <- function(primary, p_same_sign, p_min) {
  if (primary$mu_lower <= 0 && primary$mu_upper >= 0) {
    "not assessed"
  } else if (all(p_same_sign >= p_min)) {
    "robust"
  } else {
    "not robust"
  }
}

# Model 2: group x age (or group + spline of age) -------------------------------------------
model2_data <- function(df, metric, spline_df = 0) {
  controls <- df %>% filter(group == "control")
  y_mean <- mean(controls[[metric]])
  y_sd <- sd(controls[[metric]])
  age_sd <- sd(df$age)

  basis <- if (spline_df > 0) ns(df$age, df = spline_df) else matrix(0, nrow(df), 0)

  list(
    data = list(
      n = nrow(df),
      y = (df[[metric]] - y_mean) / y_sd,
      group = as.numeric(df$group == "test"),
      age = (df$age - 5) / (2 * age_sd),
      sex = ifelse(df$sex == "M", 0.5, -0.5),
      k = ncol(basis),
      b = unclass(basis)[, seq_len(ncol(basis)), drop = FALSE]
    ),
    y_mean = y_mean,
    y_sd = y_sd,
    age_sd = age_sd
  )
}

# posterior draws of the linear Model 2 (the age terms are vectors of size 1) ---------
model2_draws <- function(fit) {
  draws <- as_draws_df(fit$draws(c("alpha", "b_group", "b_age", "b_group_age")))
  tibble(
    alpha = draws$alpha,
    b_group = draws$b_group,
    b_age = draws$`b_age[1]`,
    b_group_age = draws$`b_group_age[1]`
  )
}

# posterior regression lines of Model 2 on the original scale (sex averaged) ---------
# each group's line only within the ages observed in that group
model2_lines <- function(fit, prep, df, width = 0.95) {
  draws <- model2_draws(fit)
  grid <- df %>%
    group_by(group) %>%
    reframe(age = round(seq(min(age), max(age), by = 0.1), 1))

  lines <- pmap_dfr(grid, function(group, age) {
    g <- as.numeric(group == "test")
    a <- (age - 5) / (2 * prep$age_sd)
    eta <- draws$alpha + draws$b_group * g + draws$b_age * a + draws$b_group_age * g * a
    y <- eta * prep$y_sd + prep$y_mean
    tibble(
      group = group,
      age = age,
      mean = mean(y),
      lower = unname(quantile(y, (1 - width) / 2)),
      upper = unname(quantile(y, 1 - (1 - width) / 2))
    )
  })

  lines
}

fit_model2 <- function(model, df, metric, analysis, band, spline_df = 0) {
  prep <- model2_data(df, metric, spline_df)
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

  if (spline_df == 0) {
    draws <- model2_draws(fit)
    out <- bind_rows(
      summarise_draws_vec(draws$b_group, "group (age 5)"),
      summarise_draws_vec(draws$b_age * per_year, "slope control (per year)"),
      summarise_draws_vec((draws$b_age + draws$b_group_age) * per_year, "slope test (per year)"),
      summarise_draws_vec(draws$b_group_age * per_year, "group x age (per year)")
    )
  } else {
    draws <- as_draws_df(fit$draws("b_group"))
    out <- summarise_draws_vec(draws$b_group, "group (spline age)")
  }

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
