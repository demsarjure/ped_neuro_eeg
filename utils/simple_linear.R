# libraries
library(bayesplot)
library(cmdstanr)
library(ggplot2)
library(ggdist)
library(mcmcse)
library(posterior)
library(tidyverse)

# fit the linear model ---------------------------------------------------------
fit_simple_linear <- function(x, y, robust = FALSE) {
  # load the model
  if (!robust) {
    model <- cmdstan_model("./models/linear.stan")
  } else {
    model <- cmdstan_model("./models/linear_robust.stan")
  }

  # remove NAs
  filtered_x <- x[!is.na(x) & !is.na(y)]
  filtered_y <- y[!is.na(x) & !is.na(y)]

  # prep data
  stan_data <- list(
    n = length(filtered_x),
    x = filtered_x,
    y = filtered_y
  )

  # fit
  fit <- model$sample(
    data = stan_data,
    parallel_chains = 4,
    refresh = 0
  )

  print(mcmc_trace(fit$draws()))
  print(fit$summary())

  return(fit)
}

# compare the simple_linear fit with a constant --------------------------------
compare_simple_linear <- function(fit, constant = 0) {
  # extract
  df_samples <- as_draws_df(fit$draws())

  # compare
  positive <- mcse(df_samples$b > constant)
  negative <- mcse(df_samples$b < constant)

  # extract
  positive_prob <- round(positive[[1]] * 100, 2)
  positive_se <- round(positive[[2]] * 100, 2)
  negative_prob <- round(negative[[1]] * 100, 2)
  negative_se <- round(negative[[2]] * 100, 2)

  # print results
  cat(paste0(
    "# P(b > ", constant, ") = ",
    positive_prob, " +/- ", positive_se, "%\n"
  ))
  cat(paste0(
    "# P(b < ", constant, ") = ",
    negative_prob, " +/- ", negative_se, "%\n"
  ))

  return(list(
    positive_prob = positive_prob,
    positive_se = positive_se,
    negative_prob = negative_prob,
    negative_se = negative_se
  ))
}

# plot the simple_linear model's fit -------------------------------------------
plot_simple_linear <- function(fit, data) {
  # get samples
  df_samples <- as_draws_df(fit$draws())

  # get mean averages
  a <- df_samples$a
  b <- df_samples$b

  # number of samples
  n <- length(a)

  # x min and max
  x_min <- min(data$x)
  x_max <- max(data$x)

  df <- tibble(
    draw = 1:n,
    x = list(x_min:x_max),
    y = map2(a, b, ~ .x + .y * x_min:x_max)
  ) %>% unnest(c(x, y))

  p <- df %>%
    group_by(x) %>%
    median_qi(y, .width = c(.50, .90)) %>%
    ggplot() +
    geom_lineribbon(aes(x = x, y = y, ymin = .lower, ymax = .upper), show.legend = FALSE, linewidth = 0.5) +
    geom_point(data = data, aes(x = x, y = y), color = "black", alpha=0.2, shape = 16) +
    scale_fill_brewer() +
    theme_minimal()

  return(p)
}
