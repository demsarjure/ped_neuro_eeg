data {
  int<lower=1> n; // total number of data points
  vector[n] y;    // target variable
}

parameters {
  real mu;             // mu
  real<lower=0> sigma; // stdev
}

model {
  // model
  y ~ cauchy(mu, sigma);
}
