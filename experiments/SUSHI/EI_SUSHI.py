#!/usr/bin/env python
# coding: utf-8

# # Preferential Bayesian Optimization: EI

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

sys.path.append(os.path.split(os.path.split(os.path.split(os.getcwd())[0])[0])[0]) # Move 3 levels up directory to import PBO
import PBO


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


features = pickle.load( open( "sushi_features.p", "rb" ) )


# In[ ]:


fvals = pickle.load( open( "fvals.p", "rb" ) )


# In[ ]:


# construct dict
feat_to_fval_dict = {}
for i in range(len(features)):
    key = features[i].data.tobytes()
    feat_to_fval_dict[key] = fvals[i]


# In[ ]:


objective = lambda x: PBO.objectives.sushi(x, feat_to_fval_dict)
objective_low = np.min(features)
objective_high = np.max(features)
objective_name = "SUSHI"
acquisition_name = "EI"
experiment_name = acquisition_name + "_" + objective_name


# In[ ]:


num_runs = 10
num_evals = 35
num_choices = 2
input_dims = 6
num_maximizers = 20
num_maximizers_init = 50
num_fourier_features = 1000
num_init_prefs = 10


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
    
    # Train model with data
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


# Generate rank dictionary and immediate regret dictionary.

# In[ ]:


fval_idx_tuples = pickle.load(open("fval_idx_tuples.p", "rb"))


# In[ ]:


rank_dict = {}

for i in range(len(fval_idx_tuples)):
    rank_dict[features[fval_idx_tuples[i][1]].data.tobytes()] = i + 1


# This function is our main metric for the performance of the acquisition function.

# In[ ]:


def get_max_sushi(model, features, rank_dict):
    """
    :param model: gpflow model
    :param features: sushi features
    :param rank_dict: dictionary from sushi idx to place in ranking
    :return: tuple (index of max sushi, rank)
    """
    f_preds = model.predict_f(features)[0]
    max_idx = np.argmax(f_preds)
    
    return (max_idx, rank_dict[features[max_idx].data.tobytes()])


# In[ ]:


def get_maximizer(model, data):  # For EI
    fvals = model.predict_f(data)[0]
    argmax = np.argmax(fvals)
    return data[argmax]


# Store the results in these arrays:

# In[ ]:


num_data_at_end = int(num_init_prefs + num_evals)
X_results = np.zeros([num_runs, num_data_at_end, num_choices, input_dims])
y_results = np.zeros([num_runs, num_data_at_end, 1, input_dims])
immediate_regret = np.zeros([num_runs, num_evals], np.int32)


# Create the initial values for each run:

# In[ ]:


np.random.seed(0)
random_indices = np.random.choice(features.shape[0], [num_runs, num_init_prefs, num_choices])
init_vals = np.take(features, random_indices, axis=0)


# The following loops carry out the Bayesian optimization algorithm over a number of runs, with a fixed number of evaluations per run.

# In[ ]:


for run in range(num_runs):  # CHECK IF STARTING RUN IS CORRECT
    print("Beginning run %s" % (run))
    
    X = init_vals[run]
    y = get_noisy_observation(X, objective)
    
    model, inputs, u_mean, inducing_vars = train_and_visualize(X, y, "Run_{}:_Initial_model".format(run))

    for evaluation in range(num_evals):
        print("Beginning evaluation %s" % (evaluation)) 
        
        # Get incumbent maximizer
        maximizer = np.expand_dims(get_maximizer(model, inputs), axis=0)  # (1, input_dims)
        
        success = False
        fail_count = 0
        while not success:
            # Sample possible next input points. In EI, all queries are a pair with the incumbent maximizer as the 
            # first point and a next input point as the second point
            samples = features  # (num_samples, input_dims)  # for SUSHI, small dataset so calculate over all points

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

            X_temp = np.concatenate([X, [next_query]])
            # Evaluate objective function
            y_temp = np.concatenate([y, get_noisy_observation(np.expand_dims(next_query, axis=0), objective)], axis=0)
            
            try:
                print("Evaluation %s: Training model" % (evaluation))
                model, inputs, u_mean, inducing_vars = train_and_visualize(X_temp, y_temp,
                                                                           "Run_{}_Evaluation_{}".format(run, evaluation))
                success = True

            except ValueError as err:
                print(err)
                print("Retrying sampling random inputs")
                fail_count += 1

            if fail_count >= 3:
                print("Retry limit exceeded")
                raise ValueError("Failed")
        
        X = X_temp
        y = y_temp
        
        # Save model
        pickle.dump((X, y, inputs, 
                     model.kernel.variance, 
                     model.kernel.lengthscale, 
                     model.likelihood.variance, 
                     inducing_vars, 
                     model.q_mu, 
                     model.q_sqrt), 
                    open(results_dir + "Model_Run_{}_Evaluation_{}.p".format(run, evaluation), "wb"))

        (max_idx, rank) = get_max_sushi(model, features, rank_dict)
        immediate_regret[run, evaluation] = rank - 1
        
        print("Maximizing sushi has index {} and rank {}".format(max_idx, rank)) 

    X_results[run] = X
    y_results[run] = y
    print("Run {} immediate regret: ".format(run))
    print(immediate_regret[run])


# In[ ]:


pickle.dump((X_results, y_results, immediate_regret), open(results_dir + "res.p", "wb"))


# In[ ]:


ir = immediate_regret 
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

