// Check of Model 1: matched-set model on standardised values (centred on the
// matched controls' mean and divided by the pooled within-stratum SD in R, so
// the prior on theta does not pull mu). Every stratum has its own level theta;
// test children differ from it by mu.
data {
  int<lower=1> n_control;
  int<lower=1> n_test;
  int<lower=1> n_strata;
  array[n_control] int<lower=1, upper=n_strata> stratum_control;
  array[n_test] int<lower=1, upper=n_strata> stratum_test;
  vector[n_control] y_control;
  vector[n_test] y_test;
}

parameters {
  vector[n_strata] theta;                      // stratum levels
  real mu;                                     // test minus control
  real<lower=0> sigma_control;
  real<lower=0> sigma_test;
  real<lower=1> nu;
}

model {
  // priors
  theta ~ normal(0, 2.5);
  mu ~ normal(0, 1);
  sigma_control ~ exponential(1);
  sigma_test ~ exponential(1);
  nu ~ gamma(2, 0.1);

  // model
  y_control ~ student_t(nu, theta[stratum_control], sigma_control);
  y_test ~ student_t(nu, theta[stratum_test] + mu, sigma_test);
}
