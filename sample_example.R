library(Rcpp)
library(RcppArmadillo)
library(bayesm)
library(MASS)
sourceCpp('multivariate_sv.cpp')


set.seed(10)
N_elec = J = 5
N_ev = 1
n_params = 35*seq(30,200,30)
Time = 2000 
b = matrix(runif(N_elec*N_elec, -0.05,0.05), N_elec,N_elec)
alpha = runif(N_elec, -0.3, -0.1)
diag(b) = runif(N_elec,0.9,0.95)

x0 = matrix(0,N_ev, N_elec)
x = array(0, dim = c(Time,N_ev,N_elec))
y = array(0, dim = dim(x))
log_y_squared = y

Sigma = diag(N_elec)*0.5
mix_mat =  array(0, dim = dim(x))

p= c(0.00609, 0.04775, 0.13057, 0.20674, 0.22715, 0.18842, 0.12047, 0.05591, 0.01575,  0.00115)
m = c( 1.92677, 1.34744, 0.73504,0.02266, -0.85173, -1.97278,  -3.46788, -5.55246, -8.68384, -14.65000 )
v= c(0.11265, 0.17788, 0.26768, 0.40611, 0.62699,  0.98583, 1.57469, 2.54498 , 4.16591,7.33342 )


N_p = 10
for(i in 1:N_ev)
{
  x0[i,] = rnorm(N_elec,alpha/(1-diag(b)), sqrt(diag(Sigma)/(1-diag(b)^2)))
}


for(i in 1:N_ev)
{
  
  for(t in 1:Time)
  {
    indices = sample(1:N_p,N_elec,TRUE,p)
    if(t==1)
    {
      x[t,i,]  = alpha + b%*%x0[i,] + mvrnorm(1, rep(0, N_elec), Sigma)
    }else
    {
      
      x[t,i,] = alpha + b%*%x[t-1,i,] +  mvrnorm(1, rep(0, N_elec), Sigma)
    }
    
    mix_mat[t,i,] = indices
    for(j in 1:N_elec){
      y[t,i,j] = rnorm(1, 0, exp(x[t,i,j]/2.0))}
  }
}

off_set = 1.0e-6*min(y^2)
log_y_squared_obs = log(y^2+ off_set)


y_long = matrix(y, Time*N_ev, N_elec)
x_long = matrix(x, Time*N_ev, N_elec)


mix_mat = mix_mat -1

# Gibbs sampling
n_channels = J
mu1 = rep(0,n_channels)
sigma1= diag(n_channels)*10
c = -0.5
d = 0.0
n_burnin= 1000
n_draws = 1000
n_thin = 1
n_samp = n_draws/n_thin
theta0 = rep(0.5,n_channels+1)
Sigma_theta0_inv = diag(n_channels+1)*0.001

off_set = 1.0e-6
bmu = rep(0,n_channels)
Bmu = diag(n_channels)*1000

Bsigma = diag(J)*1000

Bsigma = 1000

mu_init = rep(0,J)
b_init = diag(J)*0.8
sigsq_init = diag(Sigma)


btheta = rep(0, n_channels+1)
Btheta = diag(n_channels+1)*10^3
a0 = 1.0
b0 = 1.0

mix_mat_init = matrix(0, n_channels, Time)
x_init = matrix(runif(n_channels*(Time+1), -2,2), n_channels , Time +1)


# Run MSV model on timeseries y
pt= proc.time()
result = MSV(t(y[,1,]), p,m,v,mu1, sigma1, theta0, Sigma_theta0_inv, c, d, n_burnin, n_draws, n_thin, mu_init, b_init, sigsq_init, off_set, mix_mat_init,
             x_init, bmu, Bmu, Bsigma, btheta, Btheta, a0, b0 )
run_time = proc.time()-pt



# get posterior SV samples
quantile_vec = c(0.025,0.5,0.975)
x_samp =result$x
x_post = apply(x_samp,c(1,2),function(x){quantile(x, c(0.025,0.5,0.975))})
run_time = proc.time()-pt

# plot posterior latent SV series 
for(channel in 1:J){
  plot(x[,1,channel], type = 'l', col = 'red',main = paste('channel', as.character(channel)), xlab = 'Time', ylab = "")
  plot(x[,1,channel], type = 'l', col = 'red',main = paste('channel', as.character(channel)), xlab = 'Time', ylab = "")
  
  lines(x_post[2,channel,], col = 'blue')
  lines(x_post[3,channel,], col = 'grey')
  lines(x_post[1,channel,], col = 'grey')
  legend('topleft',legend = c("True", "MSV"), col = c("red", "blue"), lwd = c(1.5,1.5), bty = 'n')
  
}




      

