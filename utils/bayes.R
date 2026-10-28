# libraries
library(cmdstanr)
library(posterior)
library(tidyverse)

# sampling settings
SEED <- 42
CHAINS <- 4
ITER_WARMUP <- 1000
ITER_SAMPLING <- 2500
# a fit that fails the checks is refitted once with these settings
REFIT_ADAPT_DELTA <- 0.99
REFIT_MAX_TREEDEPTH <- 12

# compile a model ---------------------------------------------------------------
# compiled into a temporary directory so that no binaries are written to models/
compile_model <- function(name) {
  cmdstan_model(
    file.path("models", paste0(name, ".stan")),
    dir = tempdir(),
    quiet = TRUE
  )
}

# check convergence ---------------------------------------------------------------
# a fit fails if any sampled parameter (not transformed parameters or generated
# quantities) has R-hat >= 1.01 or a bulk or tail ESS <= 400, or if there is any
# divergence, any chain with E-BFMI < 0.3 or any transition at the maximum tree
# depth; returns the failed checks (none for a fit that passes)
check_fit <- function(model, fit) {
  pars <- intersect(names(model$variables()$parameters), fit$metadata()$stan_variables)
  summ <- fit$summary(pars, "rhat", "ess_bulk", "ess_tail")
  diag <- fit$diagnostic_summary(quiet = TRUE)

  c(
    if (sum(diag$num_divergent) > 0) paste(sum(diag$num_divergent), "divergences"),
    if (any(summ$rhat >= 1.01, na.rm = TRUE)) "R-hat >= 1.01",
    if (any(summ$ess_bulk <= 400, na.rm = TRUE)) "bulk ESS <= 400",
    if (any(summ$ess_tail <= 400, na.rm = TRUE)) "tail ESS <= 400",
    if (any(diag$ebfmi < 0.3, na.rm = TRUE)) "E-BFMI < 0.3",
    if (sum(diag$num_max_treedepth) > 0) paste(sum(diag$num_max_treedepth), "transitions at maximum tree depth")
  )
}

# sample and check convergence --------------------------------------------------
# a fit that fails any of the checks (check_fit) is refitted once with
# REFIT_ADAPT_DELTA and REFIT_MAX_TREEDEPTH; if it still fails, it is kept and
# the failed checks are stated in `problems`
sample_model <- function(model, data) {
  run <- function(adapt_delta = 0.8, max_treedepth = 10) {
    model$sample(
      data = data,
      chains = CHAINS,
      parallel_chains = CHAINS,
      iter_warmup = ITER_WARMUP,
      iter_sampling = ITER_SAMPLING,
      seed = SEED,
      adapt_delta = adapt_delta,
      max_treedepth = max_treedepth,
      refresh = 0,
      show_messages = FALSE
    )
  }

  fit <- run()
  problems <- check_fit(model, fit)
  refit <- length(problems) > 0
  if (refit) {
    message(paste0("Refitting once (", paste(problems, collapse = ", "), ")"))
    fit <- run(REFIT_ADAPT_DELTA, REFIT_MAX_TREEDEPTH)
    problems <- check_fit(model, fit)
  }
  if (length(problems) > 0) {
    warning(paste("Sampling problems:", paste(problems, collapse = ", ")))
  }

  attr(fit, "problems") <- paste(problems, collapse = "; ")
  attr(fit, "refit") <- refit
  fit
}

# summarise a parameter ------------------------------------------------------------
# posterior mean, 95% credible interval and the probability that the parameter
# is below (or above) zero, with its Monte Carlo standard error
summarise_param <- function(fit, par, scale = 1) {
  draws <- as.numeric(as_draws_matrix(fit$draws(par))) * scale
  below <- as.numeric(draws < 0)

  tibble(
    mean = mean(draws),
    lower = unname(quantile(draws, 0.025)),
    upper = unname(quantile(draws, 0.975)),
    p_below = mean(below),
    p_below_mcse = mcse_mean(below),
    n_above = sum(draws >= 0),
    n_below = sum(draws < 0),
    ess_tail = ess_tail(draws),
    problems = attr(fit, "problems"),
    refit = attr(fit, "refit")
  )
}

# format a posterior probability as a percentage ------------------------------------
# ">99.9%" when fewer than 10 draws fall on the other side
format_p <- function(p, n_other) {
  ifelse(
    n_other < 10,
    ifelse(p > 0.5, ">99.9%", "<0.1%"),
    sprintf("%.1f%%", 100 * p)
  )
}
