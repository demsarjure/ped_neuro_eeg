// Model 2: group x age regression with a Student-t likelihood and
// group-specific residual sd. The outcome is z-scored on the controls,
// group is 0/1 (control = 0), sex is -0.5/0.5 and age is centred and divided
// by 2 SD (Gelman et al. 2008). An optional basis replaces the linear age terms
// (natural spline, sensitivity analysis S5); the linear age terms then have
// size 0, so no parameter is left outside the likelihood.
data {
  int<lower=1> n;
  vector[n] y;
  vector[n] group;
  vector[n] age;
  vector[n] sex;
  int<lower=0> k;                              // spline basis columns (0 = linear age)
  matrix[n, k] b;                              // spline basis
}

parameters {
  real alpha;
  real b_group;
  vector[k == 0] b_age;                        // linear age terms (k = 0 only)
  vector[k == 0] b_group_age;
  real b_sex;
  vector[k] b_spline;
  vector<lower=0>[2] sigma;
  real<lower=1> nu;
}

transformed parameters {
  vector[n] eta = alpha + b_group * group + b_sex * sex + b * b_spline;
  if (k == 0) {
    eta += b_age[1] * age + b_group_age[1] * group .* age;
  }
}

model {
  // priors
  alpha ~ cauchy(0, 10);
  b_group ~ cauchy(0, 2.5);
  b_age ~ cauchy(0, 2.5);
  b_group_age ~ cauchy(0, 2.5);
  b_sex ~ cauchy(0, 2.5);
  b_spline ~ cauchy(0, 2.5);
  sigma ~ exponential(1);
  nu ~ gamma(2, 0.1);

  // model
  for (i in 1:n) {
    y[i] ~ student_t(nu, eta[i], sigma[group[i] > 0.5 ? 2 : 1]);
  }
}

generated quantities {
  vector[n] y_rep;
  for (i in 1:n) {
    y_rep[i] = student_t_rng(nu, eta[i], sigma[group[i] > 0.5 ? 2 : 1]);
  }
}
