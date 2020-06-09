#!/usr/bin/env python
# coding: utf-8

# # Preferential Bayesian Optimization: Multinomial Predictive Entropy Search
# This notebook demonstrates the use of the Multinomial Predictive Entropy Search (MPES) acquisition function on ordinal (preference) data.

# In[ ]:


import numpy as np
import gpflow
import tensorflow as tf
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


objective = PBO.objectives.six_hump_camel
objective_low = -1.5
objective_high = 1.5
objective_name = "SHC"
acquisition_name = "MPES"
experiment_name = acquisition_name + "_" + objective_name


# In[ ]:


num_runs = 10
num_evals = 35
num_samples = 1000
num_choices = 2
input_dims = 2
objective_dim = input_dims # CHANGE 1: require the objective dim
num_maximizers = 20
num_maximizers_init = 50
num_fourier_features = 1000
num_init_prefs = 6 # CHANGE 2: randomly initialize with some preferences

# CHANGE 1: reduce the value of delta to avoid numerical error
# as k(x,x') = sigma^2 * exp( -[(x-x')/l]^2 )
# which could be very small if l is too small
# so we define l relatively by the range of input (objective_high - objective_low)
#   It is ok for the total number of observations > the total number of possible inputs
# because there is a noise in the observation, it might require repeated observations 
# at the same input pair to improve the confidence 
num_discrete_per_dim = 40
delta = (objective_high - objective_low) / num_discrete_per_dim


# In[ ]:


results_dir = os.getcwd() + '/results/' + experiment_name + '/'

try:
    # Create target Directory
    os.makedirs(results_dir)
    print("Directory " , results_dir ,  " created ") 
except FileExistsError:
    print("Directory " , results_dir ,  " already exists")


# Plot of the SHC function (global min at at x = [0.0898, -0.7126] and x = [-0.0898, 0.7126]):

# In[ ]:


# CHANGE 4: use a discrete grid of with cells of size = delta
inputs = PBO.models.learning_fullgp.get_all_discrete_inputs(objective_low, objective_high, objective_dim, delta)
fvals = objective(inputs).reshape(num_discrete_per_dim, num_discrete_per_dim)


# In[ ]:


fig, ax = plt.subplots()
im = ax.imshow(fvals,
          interpolation="nearest",
         extent=(objective_low, objective_high, objective_low, objective_high),
         origin="lower",
         cmap="Spectral")
fig.colorbar(im, ax=ax)
plt.show()


# In[ ]:


def plot_gp(model, inducing_points, inputs, title, cmap="Spectral"):

    side = np.linspace(objective_low, objective_high, num_discrete_per_dim)
    combs = PBO.acquisitions.dts.combinations(np.expand_dims(side, axis=1))
    predictions = model.predict_y(combs)
    preds = tf.transpose(tf.reshape(predictions[0], [num_discrete_per_dim, num_discrete_per_dim]))
    variances = tf.transpose(tf.reshape(predictions[1], [num_discrete_per_dim, num_discrete_per_dim]))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(title)
    fig.set_size_inches(18.5, 6.88)
    fig.set_dpi((200))

    ax1.axis('equal')
    im1 = ax1.imshow(preds, 
                     interpolation='nearest', 
                     extent=(objective_low, objective_high, objective_low, objective_high), 
                     origin='lower', 
                     cmap=cmap)
    ax1.plot(inducing_points[:, 0], inducing_points[:, 1], 'kx', mew=2)
    ax1.plot(inputs[:, 0], inputs[:, 1], 'ko', mew=2, color='w')
    ax1.set_title("Mean")
    ax1.set_xlabel("x0")
    ax1.set_ylabel("x1")
    fig.colorbar(im1, ax=ax1)

    ax2.axis('equal')
    im2 = ax2.imshow(variances, 
                     interpolation='nearest', 
                     extent=(objective_low, objective_high, objective_low, objective_high), 
                     origin='lower', 
                     cmap=cmap)
    ax2.plot(inducing_points[:, 0], inducing_points[:, 1], 'kx', mew=2)
    ax2.plot(inputs[:, 0], inputs[:, 1], 'ko', mew=2, color='w')
    ax2.set_title("Variance")
    ax2.set_xlabel("x0")
    ax2.set_ylabel("x1")
    fig.colorbar(im2, ax=ax2)

    plt.savefig(fname=results_dir + title + ".png")
    plt.show()


# In[ ]:


def get_noisy_observation(X, objective):
    f = PBO.objectives.objective_get_f_neg(X, objective)
    return PBO.observation_model.gen_observation_from_f(X, f, 1)


# In[ ]:


def train_and_visualize(X, y, title, lengthscale_init=None, signal_variance_init=None):
    
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
    
    # Visualize model
    plot_gp(model, inducing_vars, inputs, title)
    
    return model, inputs, u_mean, inducing_vars


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

        # Sample possible next queries
        # CHANGE 10: use discrete grid
        samples = PBO.models.learning_fullgp.sample_inputs(inputs.numpy(), 
                                                        num_samples, 
                                                        num_choices, 
                                                        min_val=objective_low, 
                                                        max_val=objective_high, 
                                                        delta=delta)

        # Sample maximizers
        print("Evaluation %s: Sampling maximizers" % (evaluation))
        maximizers = PBO.fourier_features.sample_maximizers(X=inducing_vars,
                                                            count=num_maximizers,
                                                            n_init=num_maximizers_init,
                                                            D=num_fourier_features,
                                                            model=model,
                                                            min_val=objective_low,
                                                            max_val=objective_high)
        print(maximizers)

        # Calculate PES value I for each possible next query
        print("Evaluation %s: Calculating I" % (evaluation))
        I_vals = PBO.acquisitions.pes.I_batch(samples, maximizers, model)

        # Select query that maximizes I
        next_idx = np.argmax(I_vals)
        next_query = samples[next_idx]
        print("Evaluation %s: Next query is %s with I value of %s" % (evaluation, next_query, I_vals[next_idx]))

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
                     maximizers), 
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

