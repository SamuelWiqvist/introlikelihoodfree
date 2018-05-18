# abstract type
abstract type  AdaptationAlgorithm end


# typs
"""
    FixedKernel

Fixed proposal distribution.

Parameters:

* `Cov::Array{Real}` covaraince matrix
"""
type FixedKernel <: AdaptationAlgorithm
  Cov::Array{Real}
end

"""
    AMUpdate

Adaptive tuning of the proposal distribution. Sources: *A tutorial on adaptive MCMC* [https://link.springer.com/article/10.1007/s11222-008-9110-y](https://link.springer.com/article/10.1007/s11222-008-9110-y),
and *Exploring the common concepts of adaptive MCMC and Covariance Matrix Adaptation schemes* [http://drops.dagstuhl.de/opus/volltexte/2010/2813/pdf/10361.MuellerChristian.Paper.2813.pdf](http://drops.dagstuhl.de/opus/volltexte/2010/2813/pdf/10361.MuellerChristian.Paper.2813.pdf)

Parameters:

* `C_0::Array{Real}`
* `r_cov_m::Real`
* `gamma_0::Real`
* `k::Real`
* `t_0::Integer`
"""
type AMUpdate <: AdaptationAlgorithm
  C_0::Array{Real}
  r_cov_m::Real
  gamma_0::Real
  k::Real
  t_0::Integer
end

"""
    AMUpdate

Adaptive tuning of the proposal distribution. Sources: *A tutorial on adaptive MCMC* [https://link.springer.com/article/10.1007/s11222-008-9110-y](https://link.springer.com/article/10.1007/s11222-008-9110-y),
and *Exploring the common concepts of adaptive MCMC and Covariance Matrix Adaptation schemes* [http://drops.dagstuhl.de/opus/volltexte/2010/2813/pdf/10361.MuellerChristian.Paper.2813.pdf](http://drops.dagstuhl.de/opus/volltexte/2010/2813/pdf/10361.MuellerChristian.Paper.2813.pdf)

Parameters:

* `C_0::Array{Float64}`
* `r_cov_m::Float64`
* `P_star::Float64`
* `gamma_0::Float64`
* `k::Float64`
* `t_0::Int64`
"""
type AMUpdate_gen <: AdaptationAlgorithm
  C_0::Array{Float64}
  r_cov_m::Float64
  P_star::Float64
  gamma_0::Float64
  k::Float64
  t_0::Int64
end


# functions

# set up functions
function set_adaptive_alg_params(algorithm::FixedKernel, nbr_of_unknown_parameters::Integer, theta::Vector,R::Integer)

  return (algorithm.Cov, NaN)

end

function set_adaptive_alg_params(algorithm::AMUpdate, nbr_of_unknown_parameters::Integer, theta::Vector,R::Integer)

  # define m_g m_g_1
  m_g = zeros(nbr_of_unknown_parameters,1)
  m_g_1 = zeros(nbr_of_unknown_parameters,1)

  return (algorithm.C_0, algorithm.gamma_0, algorithm.k, algorithm.t_0, [algorithm.r_cov_m], m_g, m_g_1)

end


function set_adaptive_alg_params(algorithm::AMUpdate_gen, nbr_of_unknown_parameters::Int64, theta::Vector,R::Int64)

  Cov_m = algorithm.C_0
  log_r_cov_m = log(algorithm.r_cov_m)
  log_P_star = log(algorithm.P_star)
  gamma_0 = algorithm.gamma_0
  k = algorithm.k
  t_0 = algorithm.t_0
  vec_log_r_cov_m = zeros(1,R)
  vec_log_r_cov_m[1:t_0] = log_r_cov_m
  # define m_g m_g_1
  m_g = zeros(nbr_of_unknown_parameters,1)
  m_g_1 = zeros(nbr_of_unknown_parameters,1)
  #diff_a_log_P_start = 1.
  #factor = 1.
  #gain = 1.

  return (Cov_m, log_P_star, gamma_0, k, t_0, vec_log_r_cov_m, [log_r_cov_m], m_g, m_g_1)

end


# return covariance functions
function return_covariance_matrix(algorithm::FixedKernel, adaptive_update_params::Tuple,r::Integer)

  return adaptive_update_params[1]

end

function return_covariance_matrix(algorithm::AMUpdate, adaptive_update_params::Tuple,r::Integer)
    r_cov_m = adaptive_update_params[end-2][1]
    Cov_m = adaptive_update_params[1]
    return r_cov_m^2*Cov_m
end


function return_covariance_matrix(algorithm::AMUpdate_gen, adaptive_update_params::Tuple,r::Int64)
    log_r_cov_m = adaptive_update_params[end-2][1]
    Cov_m = adaptive_update_params[1]
    return exp(log_r_cov_m)^2*Cov_m
end


# print_covariance functions
function print_covariance(algorithm::FixedKernel, adaptive_update_params::Tuple,r::Integer)

  println(adaptive_update_params[1])

end

function print_covariance(algorithm::AMUpdate, adaptive_update_params::Tuple,r::Integer)
    r_cov_m = adaptive_update_params[end-2][1]
    Cov_m = adaptive_update_params[1]
    println(r_cov_m^2*Cov_m)
end


function print_covariance(algorithm::AMUpdate_gen, adaptive_update_params::Tuple,r::Int64)
    log_r_cov_m = adaptive_update_params[end-2][1]
    Cov_m = adaptive_update_params[1]
    println(exp(log_r_cov_m)^2*Cov_m)
end


# get_covariance functions
function get_covariance(algorithm::FixedKernel, adaptive_update_params::Tuple,r::Integer)

  return adaptive_update_params[1]

end

function get_covariance(algorithm::AMUpdate, adaptive_update_params::Tuple,r::Integer)
    r_cov_m = adaptive_update_params[end-2][1]
    Cov_m = adaptive_update_params[1]
    return r_cov_m^2*Cov_m
end


function get_covariance(algorithm::AMUpdate_gen, adaptive_update_params::Tuple,r::Int64)
    log_r_cov_m = adaptive_update_params[end-2][1]
    Cov_m = adaptive_update_params[1]
    return exp(log_r_cov_m)^2*Cov_m
end

# gaussian random walk functions
function gaussian_random_walk(algorithm::FixedKernel, adaptive_update_params::Tuple, Theta::Vector, r::Integer)

  return rand(MvNormal(Theta, 1.0*adaptive_update_params[1])), zeros(size(Theta))

end

function gaussian_random_walk(algorithm::AMUpdate, adaptive_update_params::Tuple, Theta::Vector, r::Integer)
  r_cov_m = adaptive_update_params[end-2][1]
  Cov_m = adaptive_update_params[1]
  return rand(MvNormal(Theta, r_cov_m^2*Cov_m)), zeros(size(Theta))
end


function gaussian_random_walk(algorithm::AMUpdate_gen, adaptive_update_params::Tuple, Theta::Vector, r::Int64)
  log_r_cov_m = adaptive_update_params[end-2][1]
  Cov_m = adaptive_update_params[1]
  return rand(MvNormal(Theta, exp(log_r_cov_m)^2*Cov_m)), zeros(size(Theta))
end

# functions for adaptation of parameters
function adaptation(algorithm::FixedKernel, adaptive_update_params::Tuple, Theta::Array, r::Integer,a_log::Real)

  # do nothing

end

function adaptation(algorithm::AMUpdate, adaptive_update_params::Tuple, Theta::Array, r::Integer,a_log::Real)

  Cov_m = adaptive_update_params[1]
  m_g = adaptive_update_params[end-1]
  m_g_1 = adaptive_update_params[end]
  k = adaptive_update_params[3]
  gamma_0 = adaptive_update_params[2]
  t_0 = adaptive_update_params[4]

  g = r-1
  if r-1 == t_0
      m_g = mean(Theta[:,1:r-1],2)
      m_g_1 = m_g + gamma_0/( (g+1)^k ) * (Theta[:,g+1] - m_g)
      adaptive_update_params[1][:] = Cov_m + ( gamma_0/( (g+1)^k ) ) * ( (Theta[:, g+1] - m_g)*(Theta[:,g+1] - m_g)'     -  Cov_m)
      adaptive_update_params[end-1][:] = m_g_1
  elseif r-1 > t_0
      m_g_1 = m_g + gamma_0/( (g+1)^k ) * (Theta[:,g+1] - m_g)
      adaptive_update_params[1][:] = Cov_m + ( gamma_0/( (g+1)^k ) ) * ( (Theta[:, g+1] - m_g)*(Theta[:,g+1] - m_g)'     -  Cov_m)
      adaptive_update_params[end-1][:] = m_g_1
  end

end

function adaptation(algorithm::AMUpdate_gen, adaptive_update_params::Tuple, Theta::Array, r::Int64, a_log::Float64)

  Cov_m = adaptive_update_params[1]
  m_g = adaptive_update_params[end-1]
  m_g_1 = adaptive_update_params[end]
  log_P_star = adaptive_update_params[2]
  log_r_cov_m = adaptive_update_params[end-2][1]
  k = adaptive_update_params[4]
  gamma_0 = adaptive_update_params[3]
  t_0 = adaptive_update_params[5]

  g = r-1;
  if r-1 >= t_0
    diff_a_log_P_start = abs(min(1, exp(a_log)) - exp(log_P_star)) #abs( min(log(1), a_log) - log_P_star)
    factor = 1 #diff_a_log_P_start/abs(log_r_cov_m)
    gain = diff_a_log_P_start/factor
    if min(1, exp(a_log)) < exp(log_P_star) #min(log(1), a_log) < log_P_star
      sign_val = -1
    else
      sign_val = 1
    end
    adaptive_update_params[end-2][1] = log_r_cov_m + sign_val * ( gamma_0/( (g+1)^k ) ) * gain
    adaptive_update_params[6][g+1] = log_r_cov_m
    if r-1 == t_0
      m_g = mean(Theta[:,1:r-1],2)
      m_g_1 = m_g + ( gamma_0/( (g+1)^k ) ) * (Theta[:,g+1] - m_g)
      adaptive_update_params[1][:] = Cov_m + ( gamma_0/( (g+1)^k ) ) * ( (Theta[:,g+1] - m_g)*(Theta[:,g+1] - m_g)'     -  Cov_m)
      adaptive_update_params[end-1][:] = m_g_1
    elseif r-1 > t_0
      m_g_1 = m_g + gamma_0/( (g+1)^k ) * (Theta[:,g+1] - m_g)
      adaptive_update_params[1][:] = Cov_m + ( gamma_0/( (g+1)^k ) ) * ( (Theta[:,g+1] - m_g)*(Theta[:,g+1] - m_g)'     -  Cov_m)
      adaptive_update_params[end-1][:] = m_g_1
    end
  end

end
