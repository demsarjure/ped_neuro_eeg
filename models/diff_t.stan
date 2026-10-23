// Model 1: per-child differences (test child minus the mean of its matched
// controls, standardised by the pooled within-stratum control SD).
// Test children of the same sex-by-age stratum share one control mean; its
// sampling error (sd 1/sqrt(m) on the standardised scale) is a shared stratum
// term u. Optional covariates enter as child-minus-control differences.
data {
  int<lower=1> n;                              // test children
  int<lower=1> n_strata;                       // sex-by-age strata
  array[n] int<lower=1, upper=n_strata> stratum;
  vector<lower=1>[n_strata] m;                 // controls per stratum
  vector[n] d;                                 // standardised differences
  int<lower=0> k;                              // number of covariates
  matrix[n, k] x;                              // covariate differences
  real<lower=0> mu_sd;                         // prior sd of mu
}

parameters {
  real mu;                                     // mean difference
  real<lower=0> sigma;                         // between-child sd
  real<lower=1> nu;                            // degrees of freedom
  vector[n_strata] z;                          // stratum effects (non-centred)
  vector[k] beta;                              // covariate effects
}

transformed parameters {
  vector[n_strata] u = z ./ sqrt(m);
}

model {
  // priors
  mu ~ normal(0, mu_sd);
  sigma ~ exponential(1);
  nu ~ gamma(2, 0.1);
  z ~ std_normal();
  beta ~ normal(0, 1);

  // model
  d ~ student_t(nu, mu + u[stratum] + x * beta, sigma);
}

generated quantities {
  vector[n] d_rep;
  for (i in 1:n) {
    d_rep[i] = student_t_rng(nu, mu + u[stratum[i]] + x[i] * beta, sigma);
  }
}
