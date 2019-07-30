/***
 Multivariate Stochastic Volatility Model using Particle Learning Methods
Model:
y_{jt} exp(x_{jt}/2) \epsilon^y_{jt}
x_{jt} = \alpha_{j} + \beta_{j} x_{j,t-1} + \epsilon^x_{jt}
where,
\epsilon^y ~ MVN(0,I)
\epsilon^x ~ MVN(0, \Sigma)


Inputs:
y: J X T matrix of first differences of EEG signals at J channels over T periods (T is typically the length of the encoding period)
mu0, Sigma0: prior mean and covariance matrix for vector of log-vols at time 0.
N: number of particles (choosen to balance trade-off between speed and accuracy, N = 5000 typically)
m, v, p are mixtrure mean, variance, and probability vectors when linearizing y (page 3 eqn 11 and 12 in the manuscript)
prior: prior for alpha and beta


Outputs:
x: J X T matrix of latent log-volatilities
\beta: posterior estimates of persistences of volatilities (in theta variable)
\alpha: posterior estimates of intercept of volatility equation (in theta variable)
\rho: average off-diagonal entries of Sigma
*/


#include <RcppArmadillo.h>
#include <math.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadilloExtensions/sample.h>


using namespace Rcpp ;
using namespace arma;
Rcpp::Environment bayesm("package:bayesm");


// Helper Functions

Rcpp::Function rwishart = bayesm["rwishart"];
//Rcpp::Function rgamma = stats["rgamma"];


inline double logacceptrateGamma(double xnew, double xold, double Bsigma) {
  return (xold-xnew)/(2*Bsigma);
}

// acceptance ratio for log normal random walk
inline double logacceptrateRW(double xnew, double xold, double Bsigma, int T, double z) {
  return .5*(T*log(xold/xnew)+(xold-xnew)/Bsigma+(1/xold-1/xnew)*z);
}

// proportion of two beta-distributions with same parameters
// (evaluated at two different points)
inline double propBeta(double x, double y, double a, double b) {
  return pow(x/y, a-1)*pow((1-x)/(1-y), b-1);
}

// full conditional non-normalized posterior log-density of the
// degrees of freedom parameter nu
inline double logdnu(double nu, double sumtau, int n) {
  return .5 * nu * (n*log(.5*nu) - sumtau) - n*lgamma(.5*nu);
}

// first derivative of logdnu
inline double dlogdnu(double nu, double sumtau, int n) {
  return .5 * (n * ( 1 + log(.5*nu) - Rf_digamma(.5*nu)) - sumtau);
}

// second derivative of logdnu
inline double ddlogdnu(double nu, int n) {
  return .25 * n * (2/nu - Rf_trigamma(.5*nu));
}

inline double logdbeta(double x, double a, double b) {
  return (a-1)*log(x)+(b-1)*log(1-x);
}


inline double logdnorm(double x, double mu = 0, double sigma = 1) {
  return -log(sigma)-((x-mu)*(x-mu)/(2*sigma*sigma));
}



NumericVector quantile(NumericVector x, NumericVector q) {
  NumericVector y = clone(x);
  std::sort(y.begin(), y.end());
  return y[x.size()*(q - 0.000000001)];
}



double abs3(double x)
{
  return std::abs(x);
}

vec rmvnorm( mat mu , mat Sigma){
  mat C = chol(Sigma) ;
  int n = mu.n_elem ;
  mat x_old = rnorm(n,0,1) ;
  mat x = trans(C)*x_old  + mu;
  return(x);
}


using namespace Rcpp;

// [[Rcpp::export]]
arma::mat mvrnormArma(int n, arma::vec mu, arma::mat sigma) {
  int ncols = sigma.n_cols;
  arma::mat Y = arma::randn(n, ncols);
  return arma::repmat(mu, 1, n).t() + Y * arma::chol(sigma);
}



double dmvnorm(mat x, mat mu, mat Sigma)
{
  double n = mu.n_elem/1.0;
  const double pi = std::atan(1.0)*4;
  double prob = -0.5*as_scalar(trans(x-mu)*inv(Sigma)*(x-mu)) - 0.5*log(det(Sigma)) - n/2.0*log(2.0*pi)  ;
  return(prob);
}


vec Igamma(int n, double a, double b)
{
  return(1.0/rgamma(n,a, 1.0/b));
}



// [[Rcpp::export]]
double loglikelihood(mat y, mat x)
{
  int J = y.n_rows;
  int Time = y.n_cols;
  double D= 0;
  for(int t = 0; t < Time; t++)
  {
    for(int j =0; j < J; j++)
    {
      D += Rf_dnorm4(y(j,t), 0, exp(x(j,t+1)/2),1);
    }
  }
  return D;
}


// [[Rcpp::export]]
double loglikelihood_post(vec y, mat x_samp)
{
  
  int Time = y.size();
  int n_samp = x_samp.n_rows;
  
  double D= 0;
  for(int i = 0; i < n_samp; i++)
  {
    for(int t = 0; t < Time; t++)
    {
      D += Rf_dnorm4(y(t), 0, exp(x_samp(i,t)/2),1);
    }
  }
  
  return D;
}


// test FFBS algo
// [[Rcpp::export]]
List FFBS(mat y, colvec alpha, mat B, vec sigsq, mat V, colvec mu0, mat sigma0, int J, int Time,
          int n_burnin, int n_draws)
{
  colvec mu = mu0; // track mean
  mat sigma = sigma0; // track covariance
  
  colvec mut = mu0;
  mat sigmat = sigma0;
  mat kt;
  mat x = zeros(J,Time+1);
  vec v_i = zeros(J);
  int index = 0;
  mat mu_save = zeros(J,Time+1);
  cube sigma_save = zeros(J,J,Time+1);
  
  mat mu_smooth = zeros(J,Time+1);
  cube sigma_smooth = zeros(J,J,Time+1);
  
  
  mat W = diagmat(sigsq);
  int N = n_burnin + n_draws;
  
  cube x_draws = zeros(J,Time+1, n_draws);
  vec llhood_vec = zeros(n_draws);
  
  
  mu_save.col(0) = mu;
  sigma_save.slice(0) = sigma;
  
  
  
  for(int n=0; n<N; n++)
  {
    
    // forward filtering
    for(int i=1; i < (Time+1);i ++)
    {
      mut = alpha + B*mu;
      sigmat = B*sigma*B.t() + W;
      kt = sigmat*inv(sigmat + V);
      mu = mut + kt*(y.col(i-1)-mut);
      sigma = sigmat - kt*(sigmat + V)*kt.t();
      x.col(i) =  mvrnormArma(1,mu,sigma);
      mu_save.col(i) = mu;
      sigma_save.slice(i) = sigma;
    }
    
    //backward sampling
    mat sigmatt;
    colvec mutt;
    mat gt;
    colvec mutT;
    mat sigmatT;
    mutT = mu_save.col(Time);
    sigmatT = sigma_save.slice(Time);
    x.col(Time) =  mvrnormArma(1,mutT, sigmatT); // sample at T
    
    mu_smooth.col(Time) = mutT;
    sigma_smooth.slice(Time) = sigmatT;
    
    
    for(int i = Time-1; i >= 0 ; i--)
    {
      sigmat = sigma_save.slice(i);
      mut = mu_save.col(i);
      
      sigmatt = B*sigmat*B.t() + W;
      mutt = alpha + B*mut;
      
      gt = sigmat*B.t()*sigmatt.i();
      mutT = mut + gt*(mutT-mutt);
      sigmatT = sigmat -gt*(sigmatt-sigmatT)*gt.t();
      x.col(i) =   mvrnormArma(1,mutT, sigmatT);
      mu_smooth.col(i) = mutT;
      sigma_smooth.slice(i) = sigmatT;
    }
    
    // Compute log-likelihood
    double llhood_temp = 0;
    for(int i = 1; i < Time; i++)
    {
      llhood_temp += dmvnorm(y.col(i), x.col(i), V) + dmvnorm(x.col(i), alpha + B*x.col(i-1), W);
    }
    
    if ( n>=n_burnin)
    {
      x_draws.slice(n-n_burnin) = x;
      llhood_vec(n-n_burnin) = llhood_temp;
    }
  
    
  }
  return List::create(Named("x") = x, Named("llhood") = llhood_vec, Named("m") = mu_save, Named("s") = mu_smooth);
}



// sample log-volatilities using the kalman filter
// [[Rcpp::export]]
List sample_vol(mat x_current, mat y_tilde, mat sig_tilde, Mat<int> mix_mat, colvec mu_mean, mat B, vec sigsq, colvec mu1, mat sigma1, int J, int Time,
                const NumericVector m, const NumericVector v)
{
  colvec mu = mu1; // track mean
  mat sigma = sigma1; // track covariance
  colvec mut = mu1;
  mat sigmat = sigma;
  mat kt;
  mat x_tilde = zeros(J,Time+1);
  mat x = zeros(J,Time+1);
  
  mat V = zeros(J,J);
  vec y_i = zeros(J);
  vec v_i = zeros(J);
  int index = 0;
  mat mu_save = zeros(J,Time+1);
  cube sigma_save = zeros(J,J,Time+1);
  mat W = diagmat(sigsq);
  double offset = 0.000001;
  mat offset_mat = offset*eye(J,J);
  double llhood_proposed = 0;
  double llhood_current = 0;
  // forward filtering
  mat y_processed = zeros(J,Time+1);
  mu_save.col(0) = mu;
  sigma_save.slice(0) = sigma;
  
  
  for(int i=1; i < (Time+1);i ++)
  {
    for(int j=0; j< J; j++)
    {
      index = mix_mat(j,i-1);
      y_i(j) = y_tilde(j,i-1) - mu_mean(j);
      v_i(j) = v(index);
    }
    
    V = diagmat(v_i);
    mut = B*mu;
    sigmat = B*sigma*B.t() + W;
    kt = sigmat*inv(sigmat + V);
    mu = mut + kt*(y_i-mut);
    sigma = sigmat - kt*(sigmat + V)*kt.t();
    x_tilde.col(i) =  mvrnormArma(1,mu,sigma).t();
    mu_save.col(i) = mu;
    sigma_save.slice(i) = sigma ;
    
  }
  
  // backward sampling
  mat sigmatt;
  colvec mutt;
  mat gt;
  colvec mutT;
  mat sigmatT;
  
  mutT = mu_save.col(Time);
  sigmatT = sigma_save.slice(Time) + eye(J,J)*0.0001  ;
  
  x_tilde.col(Time) =  mvrnormArma(1,mutT, sigmatT).t(); // sample at T
  
  for(int i = Time-1; i >= 0 ; i--)
  {
    sigmat = sigma_save.slice(i);
    mut = mu_save.col(i);
    sigmatt = B*sigmat*B.t() + W;
    mutt =  B*mut;
    gt = sigmat*B.t()*sigmatt.i();
    mutT = mut + gt*(x_tilde.col(i+1)-mutt);
    sigmatT = sigmat - gt*B*sigmat;
    x_tilde.col(i) =  mvrnormArma(1,mutT, sigmatT).t();
    x.col(i) = x_tilde.col(i) + mu_mean;
  }
  
  
  
  return List::create(Named("x")= x, Named("q_proposed") = llhood_proposed, Named("q_current") = llhood_current, Named("y") = y_processed); // return x and likelihood
}




// 1 block sampler
List sample_regression_conjugate_MH(mat x, colvec mu_old, mat B_old, vec sigsq_old, colvec theta0, mat Sigma_theta0_inv, double c, double d, colvec bmu, mat Bmu, double Bsigma,
                                    vec btheta, mat Btheta, double a0, double b0, mat y_tilde, mat sig_tilde)
{
  // sample theta = c(alpha,beta)
  int J = x.n_rows;
  int Time = x.n_cols;
  mat theta = ones(J,J+1)*0.5;
  colvec x_tt = zeros(Time-1);
  mat x_t = zeros(Time-1,J+1);
  mat sigma_theta_post;
  colvec mu_theta_post;
  mat XX;
  mat eyebeta_old = eye(J,J) - B_old;
  
  colvec alpha_old = eyebeta_old*mu_old;
  
  mat theta_old = join_rows(alpha_old, B_old);
  // sample sigsq
  double c_post = c + 0.5*(Time-1);
  double d_post = d;
  double R;
  
  vec sigsq = sigsq_old;
  colvec alpha = alpha_old;
  mat B = B_old;
  
  rowvec eyebeta = zeros(1,J);
  colvec theta_temp;
  double sigsq_prop_j;
  
  vec b_mu_prop;
  vec B_mu_prop;
  
  vec b_mu_old;
  vec B_mu_old;
  
  
  for(int j=0; j<J; j++)
  {
    // Reset R
    R = 0;
    // sample theta_j
    x_tt = x.submat(j,1,j,Time-1).t(); //
    x_t = join_rows(ones(Time-1,1),x.submat(0,0,J-1,Time-2).t());
    XX = (x_t.t()*x_t);
    sigma_theta_post = (Sigma_theta0_inv + XX).i();
    mu_theta_post = sigma_theta_post*(Sigma_theta0_inv*theta0 + x_t.t()*x_tt);
    
    //d_post = d + 0.5*as_scalar(theta0.t()*Sigma_theta0_inv*theta0 + dot(x_tt,x_tt) - mu_theta_post.t()*sigma_theta_post.i()*mu_theta_post);
    // get proposed values for alpha, beta, sigma
    d_post = d + 0.5*as_scalar(theta0.t()*Sigma_theta0_inv*theta0 + dot(x_tt,x_tt) - mu_theta_post.t()*x_t.t()*x_tt);
    sigsq_prop_j = Igamma(1,c_post, d_post)(0);
    
    theta_temp = rmvnorm(mu_theta_post, sigma_theta_post*sigsq_prop_j);
    
    
    
    for(int k = 0; k < J; k++)
    {
      if( (k) == j)
      {
        eyebeta(k) = 1.0-theta_temp(k+1);
      }
      else
      {
        eyebeta(k) = -theta_temp(k+1);
      }
      
    }
    
    R = logacceptrateGamma(sigsq_prop_j, sigsq_old(j), Bsigma);
    
    // Rcout << "diff sigsq " << R  << endl;
    // metropolis hasting step
    
    
    for(int k = 0; k < J; k++)
    {
      R += logdbeta((theta_temp(k+1) + 1)/2, a0, b0);
      R -= logdbeta((B_old(j,k) + 1)/2, a0, b0);
    }

  
    b_mu_prop = eyebeta*bmu;
    B_mu_prop = eyebeta*Bmu*eyebeta.t();
    b_mu_old = eyebeta_old.row(j)*bmu;
    B_mu_old = eyebeta_old.row(j)*Bmu*eyebeta_old.row(j).t();
    R += logdnorm(theta_temp(0), b_mu_prop(0), sqrt(B_mu_prop(0)));
    R -= logdnorm(alpha_old(j), b_mu_old(0), sqrt(B_mu_old(0)));

    
    R += dmvnorm(theta_temp, btheta, sigsq_prop_j*Btheta);
    R -= dmvnorm(theta_old.row(j).t(), btheta, sigsq_old(j)*Btheta);
    
    
    if(log(runif(1)(0)) < R)
    {
      theta.row(j) = theta_temp.t();
      sigsq(j) = sigsq_prop_j;
    }else
    {
      theta.row(j) = theta_old.row(j);
      sigsq(j) = sigsq_old(j);
    }
  }
  
  
  B = theta.submat(0,1,J-1,J);
  alpha = theta.col(0);
  vec mu = (eye(J,J)-B).i()*alpha;
  
  
  
  for(int i = 0; i < J; i++)
  {
    for(int j = 0; j < J; j++)
    {
      if(i!=j)
      {
        if(abs3(B(i,j)) >= abs3(B(i,i)))
        {
          B(i,j) = B_old(i,j);
        }
      }
    }
    if((B(i,i) < 0.0) || B(i,i) > 1.0)
    {
      B(i,i) = B_old(i,i);
    }
  }
  
  
  // centered
  mat x_tilde = x;
  for(int j = 0; j < J; j++)
  {
    x_tilde.row(j) = (x.row(j)-mu(j))/sqrt(sigsq(j));
  }
  
  mat x_tilde_t;
  colvec sig_t;
  mat sigma_omega_post;
  mat mu_omega_post;
  colvec y_tilde_t;
  rowvec omega;
  
  
  for(int j = 0; j < J; j++)
  {
    x_tt = x_tilde.submat(j,1,j,Time-1).t();
    // x_tt = x_tt;
    // draw mu and sigma
    y_tilde_t = y_tilde.row(j).t();
    sig_t = sig_tilde.row(j).t();
    y_tilde_t = y_tilde_t/sig_t;
    x_tilde_t = join_rows(ones(Time-1,1)/sig_t, x_tt/sig_t);
    
    //Rcout << x_tilde_t.t()*x_tilde_t << endl;
    sigma_omega_post = (eye(2,2)*0.0001 + x_tilde_t.t()*x_tilde_t).i();
    mu_omega_post = sigma_omega_post*(x_tilde_t.t()*y_tilde_t);
    omega = rmvnorm(mu_omega_post, sigma_omega_post).t();
    mu(j) = omega(0);
    sigsq(j) = omega(1)*omega(1);
  }
  
  for(int t = 0; t < Time; t++)
  {
    for(int j = 0; j<J; j++){
      x(j,t) = mu(j) + x_tilde(j,t)*sqrt(sigsq(j));
    }
  }
  
  return List::create(Named("x") = x, Named("mu") = mu, Named("B") = B, Named("sigsq") = sigsq);
}


// sample mixture indices and return y_star
// [[Rcpp::export]]
Mat<int> sample_mix_index(mat log_y_squared, mat x, const NumericVector p, const NumericVector m, const NumericVector v,  int n_mixture, double offset)
{
  int J = log_y_squared.n_rows;
  int Time = log_y_squared.n_cols;
  Mat<int> mix_mat(J,Time);
  NumericVector probs(n_mixture);
  IntegerVector ind = seq_len(n_mixture)-1;
  int index = 0;
  
  for(int i=1; i < (Time+1); i++)
  {
    for(int j=0; j < J; j++)
    {
      for(int k = 0; k < n_mixture; k++)
      {
        probs(k) = Rf_dnorm4(log_y_squared(j,i-1), m(k) + x(j,i), sqrt(v(k)),0)*(p(k));
      }
      //probs = exp(probs)/sum(exp(probs)); // normalize
      probs = probs/sum(probs);
      index = Rcpp::RcppArmadillo::sample(ind,1,true,probs)(0);
      mix_mat(j,i-1) = index;
    }
  }
  return mix_mat;
}



Mat<int> sample_mix_index_multivariate(mat log_y_squared, mat x, const NumericVector p, const NumericVector m, const NumericVector v,  int n_mixture, double offset, vec sigsq, mat B, colvec mu)
{
  int J = log_y_squared.n_rows;
  int Time = log_y_squared.n_cols;
  Mat<int> mix_mat(J,Time);
  NumericVector probs(n_mixture);
  IntegerVector ind = seq_len(n_mixture)-1;
  int index = 0;
  
  mat W = diagmat(sigsq);
  mat V_temp = eye(J,J);
  colvec m_vec = zeros(J);
  
  
  for(int i=1; i < (Time+1); i++)
  {
    for(int k=0; k < n_mixture; k++)
    {
      for(int j = 0;j < J;j++)
      {
        V_temp(j,j)= v[k];
        m_vec(j) = m[k];
      }
      probs[k] = dmvnorm(log_y_squared.col(i-1), m_vec + mu +  B*(x.col(i-1)-mu), V_temp + W) + log(p(k));
    }
    probs = exp(probs)/sum(exp(probs)); // normalize
    //probs = probs/sum(probs);
    index = Rcpp::RcppArmadillo::sample(ind,1,true,probs)(0);
    for(int j =0;j< J;j++){
      mix_mat(j,i-1) = index;}
    
  }
  return mix_mat;
}


// main function
// [[Rcpp::export]]
List MSV(mat y, const NumericVector p, const NumericVector m, const NumericVector v, colvec mu1, mat sigma1, colvec theta0, mat Sigma_theta0_inv, double c, double d, int n_burnin, int N_draws, int n_thin, colvec mu_init, mat B_init,
         vec sigsq_init, double offset, Mat<int> mix_mat_init, mat x_init, colvec bmu, mat Bmu, double Bsigma, vec btheta, mat Btheta, double a0, double b0)
{
  
  int n_draws = N_draws/n_thin;
  
  int N = n_burnin + N_draws; // number for iterations
  int J = y.n_rows; // number of elecs
  int Time = y.n_cols; // number of time periods
  mat B = B_init; // persistence matrix
  colvec mu = mu_init; // mean
  colvec alpha = zeros(J); // intercept
  
  vec sigsq = zeros(J); // vector of variances of shocks
  mat V = zeros(J,J);
  mat W = zeros(J,J);
  mat y_star = zeros(J,J);
  
  // stores draws
  cube x_draws = zeros(J,Time+1,n_draws);
  mat mu_draws = zeros(J,n_draws);
  cube B_draws = zeros(J,J, n_draws);
  mat sigsq_draws = zeros(J, n_draws);
  Cube<int> mix_mat_draws(J,Time, n_draws);
  
  int n_mixture = p.length();
  
  double D_bar = 0;
  double harmonic = 0;
  
  
  sigsq = sigsq_init;
  
  mat log_y_squared = log(square(y) + offset);
  Mat<int> mix_mat ;
  mix_mat = mix_mat_init;
  mat x_i_current = x_init;
  double q_i_proposed = 0;
  double q_i_current = 0;
  
  

  mat theta = zeros(J, J+1);
  for(int j=0;j < J; j++)
  {
    theta(j,j+1)= 0.5;
  }
  
  List result_proposed;
  List result_regression;
  
  
  mat y_tilde = log_y_squared;
  mat sig_tilde = zeros(J,Time);
  int index = 0;
  
  
  for(int i=0; i < N;  i++)
  {
    // sample indicator matrix
    //mix_mat = sample_mix_multivariate(log_y_squared, x_i_current,p,m,v,n_mixture,offset,sigsq, B, mu);
    mix_mat = sample_mix_index(log_y_squared, x_i_current,p,m,v,n_mixture,offset);
    //
    //
    //
    for(int i=0; i < Time;i ++)
    {
      for(int j=0; j< J; j++)
      {
        index = mix_mat(j,i);
        y_tilde(j,i) = log_y_squared(j,i) - m(index) ;
        sig_tilde(j,i) = sqrt(v(index));
      }
    }
    
    result_proposed  = sample_vol(x_i_current,y_tilde, sig_tilde, mix_mat, mu, B, sigsq, mu1, sigma1,J,Time,m,v);
    x_i_current = as<mat>(result_proposed["x"]);

    
    result_regression =  sample_regression_conjugate_MH(x_i_current, mu, B, sigsq, theta0, Sigma_theta0_inv, c, d, bmu, Bmu, Bsigma, btheta, Btheta, a0, b0, y_tilde, sig_tilde);
    B = as<mat>(result_regression["B"]) ;
    x_i_current = as<mat>(result_regression["x"]) ;
    mu = as<vec>(result_regression["mu"]) ;
    sigsq = as<vec>(result_regression["sigsq"]);
    
    if (i >= n_burnin)
    {
      if((i-n_burnin)%n_thin ==0)
      {
        int index = (i-n_burnin)/n_thin;
        
        
        x_draws.slice(index) = x_i_current;
        mu_draws.col(index) = mu;
        B_draws.slice(index) = B;
        sigsq_draws.col(index) = sigsq;
        mix_mat_draws.slice(index) = mix_mat;
        double llhood = loglikelihood(y,x_i_current);
        D_bar += llhood;
      }
    }
    
    if(i%200 ==0)
    {
      Rcout << "iteration: " << i << endl;
    }
  }
  
  return List::create(Named("x") = x_draws, Named("mu") = mu_draws, Named("B") = B_draws, Named("sigsq") = sigsq_draws,
                      Named("mix_mat") = mix_mat_draws, Named("D_bar") = D_bar, Named("harmonic") = harmonic);
  
}
