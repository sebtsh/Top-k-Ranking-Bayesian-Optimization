#!/usr/bin/env python
# coding: utf-8

# # Preferential Bayesian Optimization: EI
# This notebook demonstrates the use of the Expected Improvement (EI) acquisition function on ordinal (preference) data.

# In[ ]:


import numpy as np
import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import sys
import os
import pickle

from gpflow.utilities import set_trainable, print_summary
gpflow.config.set_default_summary_fmt("notebook")

sys.path.append(os.path.split(os.path.split(os.path.split(os.getcwd())[0])[0])[0]) # Move 3 levels up directory to import project files as module
import importlib
PBO = importlib.import_module("Top-k-Ranking-Bayesian-Optimization")


# In[ ]:


gpu_to_use = 0

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        for gpu in gpus:
              tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[gpu_to_use], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


# In[ ]:


lengthscale = 0.3
lengthscale_prior_alpha = tf.constant(2, dtype=tf.float64)
lengthscale_prior_beta = tf.constant(4, dtype=tf.float64)


# In[ ]:


objective = PBO.objectives.hartmann3d
objective_low = 0
objective_high = 1.
objective_name = "Hart3"
acquisition_name = "EI"
experiment_name = acquisition_name + "_" + objective_name


# In[ ]:


num_runs = 10
num_evals = 50
num_samples = 1000
num_choices = 2
input_dims = 3
objective_dim = input_dims # CHANGE 1: require the objective dim
num_maximizers = 20
num_init_prefs = 12 # CHANGE 2: randomly initialize with some preferences

# CHANGE 1: reduce the value of delta to avoid numerical error
# as k(x,x') = sigma^2 * exp( -[(x-x')/l]^2 )
# which could be very small if l is too small
# so we define l relatively by the range of input (objective_high - objective_low)
#   It is ok for the total number of observations > the total number of possible inputs
# because there is a noise in the observation, it might require repeated observations 
# at the same input pair to improve the confidence 
num_discrete_per_dim = 20
delta = (objective_high - objective_low) / num_discrete_per_dim


# In[ ]:


results_dir = os.getcwd() + '/results/' + experiment_name + '/'

try:
    # Create target Directory
    os.makedirs(results_dir)
    print("Directory " , results_dir ,  " created ") 
except FileExistsError:
    print("Directory " , results_dir ,  " already exists")


# In[ ]:


def get_noisy_observation(X, objective):
    f = PBO.objectives.objective_get_f_neg(X, objective)
    return PBO.observation_model.gen_observation_from_f(X, f, 1)


# In[ ]:


def train_and_visualize(X, y, title, lengthscale_init=None, signal_variance_init=None):
    lengthscale_prior = tfp.distributions.Gamma(concentration=lengthscale_prior_alpha,
                                               rate=lengthscale_prior_beta)
    
    # Train model with data
    # CHANGE 6: use full_gp instead of sparse, 
    result = PBO.models.learning_fullgp.train_model_fullcov(
                        X, y, 
                        obj_low=objective_low,
                        obj_high=objective_high,
                        lengthscale_init=lengthscale_init,
                        signal_variance_init=signal_variance_init,
                        indifference_threshold=0.,
                        n_sample=1000,
                        deterministic=True, # only sample f values once, not re-sampling
                        num_steps=3000)
    
    q_mu = result['q_mu']
    q_sqrt = result['q_sqrt']
    u = result['u']
    inputs = result['inputs']
    k = result['kernel']
    
    likelihood = gpflow.likelihoods.Gaussian()
    model = PBO.models.learning.init_SVGP_fullcov(q_mu, q_sqrt, u, k, likelihood)
    u_mean = q_mu.numpy()
    inducing_vars = u.numpy()
    
    return model, inputs, u_mean, inducing_vars


# In[ ]:


def uniform_grid(input_dims, num_discrete_per_dim, low=0., high=1.):
    """
    Returns an array with all possible permutations of discrete values in input_dims number of dimensions.
    :param input_dims: int
    :param num_discrete_per_dim: int
    :param low: int
    :param high: int
    :return: tensor of shape (num_discrete_per_dim ** input_dims, input_dims)
    """
    num_points = num_discrete_per_dim ** input_dims
    out = np.zeros([num_points, input_dims])
    discrete_points = np.linspace(low, high, num_discrete_per_dim)
    for i in range(num_points):
        for dim in range(input_dims):
            val = num_discrete_per_dim ** (dim)
            out[i, dim] = discrete_points[int((i // val) % num_discrete_per_dim)]
    return out


# This function is our main metric for the performance of the acquisition function: The closer the model's best guess to the global minimum, the better.

# In[ ]:


def best_guess(model):
    """
    Returns a GP model's best guess of the global maximum of f.
    """
    # CHANGE 7: use a discrete grid
    xx = PBO.models.learning_fullgp.get_all_discrete_inputs(objective_low, objective_high, objective_dim, delta)
    res = model.predict_f(xx)[0].numpy()
    return xx[np.argmax(res)]


# Store the results in these arrays:

# In[ ]:


num_data_at_end = int(num_init_prefs + num_evals)
X_results = np.zeros([num_runs, num_data_at_end, num_choices, input_dims])
y_results = np.zeros([num_runs, num_data_at_end, 1, input_dims])
best_guess_results = np.zeros([num_runs, num_evals, input_dims])


# Create the initial values for each run:

# In[ ]:


np.random.seed(0)

# CHANGE 8: just randomly initialize with some preference observation
init_vals = np.zeros([num_runs, num_init_prefs, num_choices, input_dims])

for run in range(num_runs):
    for i in range(num_init_prefs):
        init_vals[run,i] = PBO.models.learning_fullgp.get_random_inputs(
                                objective_low, 
                                objective_high, 
                                objective_dim, 
                                delta,
                                size=num_choices,
                                with_replacement=False,
                                exclude_inputs=None)


# The following loops carry out the Bayesian optimization algorithm over a number of runs, with a fixed number of evaluations per run.

# In[ ]:


# CHANGE 9: need to store lengthscale and signal_variance from previous iteration to initialize the current iteration
lengthscale_init = None
signal_variance_init = None

for run in range(num_runs):  # CHECK IF STARTING RUN IS CORRECT
    print("")
    print("==================")
    print("Beginning run %s" % (run))
    
    X = init_vals[run]
    y = get_noisy_observation(X, objective)
    
    model, inputs, u_mean, inducing_vars = train_and_visualize(X, y, 
                                                        "Run_{}:_Initial_model".format(run))
    # save optimized lengthscale and signal variance for next iteration
    lengthscale_init = model.kernel.lengthscale.numpy()
    signal_variance_init = model.kernel.variance.numpy()
    
    for evaluation in range(num_evals):
        print("Beginning evaluation %s" % (evaluation)) 

        # Get incumbent maximizer
        input_vals = model.predict_f(inputs)[0].numpy()
        maximizer = np.expand_dims(inputs[np.argmax(input_vals)], axis=0)
        
        print("Maximizer:")
        print(maximizer)
        
        # Sample possible next input points. In EI, all queries are a pair with the incumbent maximizer as the 
        # first point and a next input point as the second point
        
        samples = PBO.models.learning_fullgp.get_random_inputs(low=objective_low,
                                                               high=objective_high,
                                                               dim=objective_dim,
                                                               delta=delta,
                                                               size=num_samples,
                                                               exclude_inputs=maximizer)

        
        # Calculate EI vals
        ei_vals = PBO.acquisitions.ei.EI(model, maximizer, samples)
        
        L = np.argsort(np.ravel(-ei_vals))  # n-th element in this (num_samples, ) size array is the index of n-th
        #largest element in ei_vals
        
        # Select query that maximizes EI
        if np.all(np.equal(samples[L[0]], maximizer)):  #if value with highest EI is same as maximizer, pick the next
            # highest value. Else pick this
            next_idx = L[1]
        else:
            next_idx = L[0]
            
        next_query = np.zeros((num_choices, input_dims))
        next_query[0, :] = maximizer  # EI only works in binary choices
        next_query[1, :] = samples[next_idx]
        print("Evaluation %s: Next query is %s with EI value of %s" % (evaluation, next_query, ei_vals[next_idx]))

        X = np.concatenate([X, [next_query]])
        # Evaluate objective function
        y = np.concatenate([y, get_noisy_observation(np.expand_dims(next_query, axis=0), objective)], axis=0)
        
        print("Evaluation %s: Training model" % (evaluation))
        model, inputs, u_mean, inducing_vars = train_and_visualize(X, y,  
                                                                   "Run_{}_Evaluation_{}".format(run, evaluation))
        
        print_summary(model)

        # save optimized lengthscale and signal variance for next iteration
        lengthscale_init = model.kernel.lengthscale.numpy()
        signal_variance_init = model.kernel.variance.numpy()

        best_guess_results[run, evaluation, :] = best_guess(model)
        # CHANGE 11: log both the estimated minimizer and its objective value
        print("Best_guess f({}) = {}".format(
                best_guess_results[run, evaluation, :], 
                objective(best_guess_results[run, evaluation, :])))
        
        # Save model
        pickle.dump((X, y, inputs, 
                     model.kernel.variance, 
                     model.kernel.lengthscale, 
                     model.likelihood.variance, 
                     inducing_vars, 
                     model.q_mu, 
                     model.q_sqrt, 
                     maximizer), 
                    open(results_dir + "Model_Run_{}_Evaluation_{}.p".format(run, evaluation), "wb"))

    X_results[run] = X
    y_results[run] = y


# In[ ]:


pickle.dump((X_results, y_results, best_guess_results), 
            open(results_dir + acquisition_name + "_" + objective_name + "_" + "Xybestguess.p", "wb"))


# In[ ]:


global_min = np.min(objective(PBO.models.learning_fullgp.get_all_discrete_inputs(objective_low, objective_high, objective_dim, delta)))
metric = best_guess_results
ir = objective(metric) - global_min
mean = np.mean(ir, axis=0)
std_dev = np.std(ir, axis=0)
std_err = std_dev / np.sqrt(ir.shape[0])


# In[ ]:


print("Mean immediate regret at each evaluation averaged across all runs:")
print(mean)


# In[ ]:


print("Standard error of immediate regret at each evaluation averaged across all runs:")
print(std_err)


# In[ ]:


with open(results_dir + acquisition_name + "_" + objective_name + "_" + "mean_sem" + ".txt", "w") as text_file:
    print("Mean immediate regret at each evaluation averaged across all runs:", file=text_file)
    print(mean, file=text_file)
    print("Standard error of immediate regret at each evaluation averaged across all runs:", file=text_file)
    print(std_err, file=text_file)


# In[ ]:


pickle.dump((mean, std_err), open(results_dir + acquisition_name + "_" + objective_name + "_" + "mean_sem.p", "wb"))

