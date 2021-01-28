#!/usr/bin/env python
# coding: utf-8

# # Preferential Bayesian Optimization: Dueling-Thompson Sampling
# 
# Implementation of the algorithm by Gonzalez et al (2017).

# In[ ]:


import numpy as np
import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import sys
import os
import datetime
import pickle

from gpflow.utilities import set_trainable, print_summary
gpflow.config.set_default_summary_fmt("notebook")

sys.path.append(os.path.split(os.path.split(os.path.split(os.getcwd())[0])[0])[0]) # Move 3 levels up directory to import project files as module
import importlib
PBO = importlib.import_module("Top-k-Ranking-Bayesian-Optimization")


# In[ ]:


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


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


def log(message):
    print(str(datetime.datetime.now()) + ': ' + message)


# In[ ]:


objective = PBO.objectives.forrester
objective_low = 0.
objective_high = 1.
objective_name = "Forrester"
acquisition_name = "DTS"
experiment_name = acquisition_name + "_" + objective_name


# In[ ]:


num_runs = 10
num_evals = 20
num_samples = 100
num_choices = 2
input_dims = 1
num_maximizers = 20
num_init_points = 3
num_inducing_init = 3
num_discrete_per_dim = 1000
num_fourier_features = 200


# In[ ]:


results_dir = os.getcwd() + '/results/' + experiment_name + '/'

try:
    # Create target Directory
    os.makedirs(results_dir)
    print("Directory " , results_dir ,  " created ") 
except FileExistsError:
    print("Directory " , results_dir ,  " already exists")


# In[ ]:


def visualize_model(query_points, y, m, title="Model", cmap="Spectral"):
    if query_points.shape[-1] != 2:
        return

    pos_vals = []
    neg_vals = []
    for i in range(len(y)):
        if y[i]:
            pos_vals.append(query_points[i])
        else:
            neg_vals.append(query_points[i])
    pos_vals = np.array(pos_vals)
    neg_vals = np.array(neg_vals)

    num_discrete_points = num_discrete_per_dim
    side = np.linspace(0, 1, num_discrete_points)
    X,Y = np.meshgrid(side,side)
    preds = tf.transpose(tf.reshape(m.predict_y(combs)[0], [num_discrete_points, num_discrete_points]))
    variances = tf.transpose(tf.reshape(PBO.acquisitions.dts.variance_logistic_f(m, combs), [num_discrete_points, num_discrete_points]))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(title)
    fig.set_size_inches(18.5, 6.88)
    fig.set_dpi((200))
    
    ax1.axis('equal')
    if len(pos_vals) != 0:
        ax1.scatter(*pos_vals.T, c="black", marker="o")
    if len(neg_vals) != 0:
        ax1.scatter(*neg_vals.T, c="black", marker="x")
    im1 = ax1.imshow(preds, interpolation='nearest', extent=(0.0, 1.0, 0.0, 1.0), origin='lower', cmap=cmap)
    ax1.set_title("Mean of y(x, x')")
    ax1.set_xlabel("x")
    ax1.set_ylabel("x'")
    ax1.axvline(x=0.757, linestyle='--')
    fig.colorbar(im1, ax=ax1)

    ax2.axis('equal')
    if len(pos_vals) != 0:
        ax2.scatter(*pos_vals.T, c="black", marker="o")
    if len(neg_vals) != 0:
        ax2.scatter(*neg_vals.T, c="black", marker="x")
    im2 = ax2.imshow(variances, interpolation='nearest', extent=(0.0, 1.0, 0.0, 1.0), origin='lower', cmap=cmap)
    ax2.set_title("Variance of y(x, x')")
    ax2.set_xlabel("x")
    ax2.set_ylabel("x'")
    fig.colorbar(im2, ax=ax2)
    
    plt.savefig(fname=results_dir + title + ".png")

    plt.show()


# In[ ]:


def visualize_f_sample(f_vals, cmap="Spectral"):
    fig, (ax1) = plt.subplots(1)
    fig.suptitle('Sampled f values')
    fig.set_size_inches(4, 3.3)
    fig.set_dpi((100))
    
    ax1.axis('equal')
    im1 = ax1.imshow(tf.transpose(tf.reshape(f_vals, [num_discrete_points, num_discrete_points])),
                     interpolation='nearest', extent=(0.0, 1.0, 0.0, 1.0), origin='lower', cmap=cmap)
    ax1.set_xlabel("x")
    ax1.set_ylabel("x'")
    ax1.axvline(x=0.757, linestyle='--')
    fig.colorbar(im1, ax=ax1)


# In[ ]:


def std_representation(X, num_choices):
    """
    :param X: tensor of shape (num_data, input_dims * num_choices)
    :return: tensor of shape (num_data, num_choices, input_dims)
    """
    input_dims = X.shape[-1] // num_choices
    ret_val = np.zeros((X.shape[0], num_choices, input_dims))
    
    for i in range(num_choices):
        ret_val[:, i, :] = X[:, input_dims*i:input_dims*(i+1)]
        
    return ret_val


# In[ ]:


def get_noisy_observation_dts(X, objective):
    """
    :param X: tensor of shape (num_data, input_dims * 2)
    :param objective: objective function
    """
    num_data = X.shape[0]
    X_std = std_representation(X, num_choices) # (num_data, num_choices, input_dims)
    f = PBO.objectives.objective_get_f_neg(X_std, objective)
    obs = np.array(PBO.observation_model.gen_observation_from_f(X_std, f, 1))  # (num_data, 1, input_dims)

    ret_val = np.zeros((num_data, 1), dtype=np.int8)
    for i in range(num_data):
        if np.allclose(X_std[i, 0], obs[i, 0]):
            ret_val[i] = 1
    return ret_val


# In[ ]:


regularizer_lengthscale_mean_over_range = 0.2
regularizer_lengthscale_std_over_range = 0.5
input_range = objective_high - objective_low
lengthscale_mean_regularizer = input_range * regularizer_lengthscale_mean_over_range
lengthscale_std_regularizer = input_range * regularizer_lengthscale_std_over_range
lengthscale = lengthscale_mean_regularizer


# In[ ]:


@tf.function
def lengthscale_regularizer(kernel):  # for product kernel
    loss = 0
    for k in kernel.kernels:
        loss += 0.5 * tf.reduce_sum(tf.square((k.lengthscale - lengthscale_mean_regularizer) / lengthscale_std_regularizer))
    
    return loss


# In[ ]:


def train_and_visualize(X, y, lengthscale, title, num_steps=3000):
    kernel = gpflow.kernels.Product([gpflow.kernels.RBF(lengthscale=lengthscale, 
                                                        active_dims=[i, i+input_dims]) 
                                     for i in range(input_dims)])
    
    m = gpflow.models.SVGP(kernel=kernel,
                           likelihood=gpflow.likelihoods.Bernoulli(invlink=tf.math.sigmoid),
                           inducing_variable=X,
                           whiten=False)
    
    m.inducing_variable.Z.trainable = False
    
    optimizer = tf.keras.optimizers.RMSprop(rho=0.0)
    
    loss = lambda: -m.log_likelihood(X, y) + lengthscale_regularizer(m.kernel)
    prev_loss = loss().numpy()
    
    for i in range(num_steps):
        optimizer.minimize(loss, m.trainable_variables)
        current_loss = loss().numpy()
        if i % 500 == 0:
            print('Loss at step %s: %s' % (i, current_loss))
        if abs((current_loss-prev_loss) / prev_loss) < 1e-7:
            print('Loss at step %s: %s' % (i, current_loss))
            break
        prev_loss = current_loss
    
    visualize_model(X, y, m, title=title)
    return m


# In[ ]:


def uniform_grid(input_dims, num_discrete_per_dim, low, high):
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


# In[ ]:


def best_guess(m, discrete_space, combs):
    return PBO.acquisitions.dts.soft_copeland_maximizer(m.predict_y(combs)[0], discrete_space)


# In[ ]:


def flip(X):
    """
    :param X: tensor of shape (num_data, input_dims * 2)
    :return: tensor of shape (num_data, input_dims * 2), where the first input_dims is swapped with the second
    """
    input_dims = X.shape[-1] // 2
    ret_val = np.zeros((X.shape))
    for i in range(X.shape[0]):
        ret_val[i, :input_dims] = X[i, input_dims:]
        ret_val[i, input_dims:] = X[i, :input_dims]
    return ret_val


# In[ ]:


def flip_y(y):
    """
    :param y: tensor of shape (num_data, 1), with int values either 0 or 1
    """
    return (y + 1) % 2


# Create the initial values for each run:

# In[ ]:


np.random.seed(0)
init_points = np.random.uniform(low=objective_low, high=objective_high, size=[num_runs, num_init_points, input_dims])
num_combs = int((num_init_points-1) * num_init_points / 2)
init_vals = np.zeros([num_runs, num_combs, num_choices, input_dims])
for run in range(num_runs):
    cur_idx = 0
    for init_point in range(num_init_points-1):
        for next_point in range(init_point+1, num_init_points):
            init_vals[run, cur_idx, 0] = init_points[run, init_point]
            init_vals[run, cur_idx, 1] = init_points[run, next_point]
            cur_idx += 1

init_vals = np.reshape(init_vals, [num_runs, num_combs, num_choices * input_dims])


# In[ ]:


discrete_space = uniform_grid(input_dims, num_discrete_per_dim, objective_low, objective_high)
combs = PBO.acquisitions.dts.combinations(discrete_space)


# Store the results in these arrays:

# In[ ]:


num_data_at_end = (num_combs + num_evals) * 2
X_results = np.zeros([num_runs, num_data_at_end, input_dims * num_choices])
y_results = np.zeros([num_runs, num_data_at_end, 1])
best_guess_results = np.zeros([num_runs, num_evals, input_dims])


# In[ ]:


for run in range(num_runs):
    #Fit a GP with kernel k to Dn
    
    X = init_vals[run]
    y = get_noisy_observation_dts(X, objective)
    
    X = np.vstack([X, flip(X)])
    y = np.vstack([y, flip_y(y)])
    
    model = train_and_visualize(X, y, lengthscale=lengthscale, title="Run_{}_Initial_model".format(run))  #TODO: CHECK LENGTHSCALE
    
    for evaluation in range(num_evals):
        log("Starting evaluation " + str(evaluation))
        
        # Sample f using RFF
        f_vals = PBO.acquisitions.dts.sample_f(model, X, combs, num_fourier_features)

        # 2 and 3. Compute the acquisition for duels alpha and get next duel
        log("Computing acquisition function")
        x_next = PBO.acquisitions.dts.soft_copeland_maximizer(f_vals, discrete_space)        
        
        all_pairs = np.concatenate([np.tile(x_next, (discrete_space.shape[0], 1)), discrete_space], axis=1)
        next_vars = np.squeeze(PBO.acquisitions.dts.variance_logistic_f(model, all_pairs), 
                               axis=1)
        xprime_next = discrete_space[np.argmax(next_vars)]
        
        x_xprime_next = np.expand_dims(np.concatenate([x_next, xprime_next]), axis=0)
        
        # Change by random small values otherwise Fourier features matrix becomes non-invertible
        if np.all(np.equal(x_xprime_next, flip(x_xprime_next))) or x_xprime_next in X:
            for i in range(len(x_xprime_next[0])):
                if x_xprime_next[0][i] < 0:
                    x_xprime_next[0][i] += np.random.uniform(low=0., high=1e-3)
                else:
                    x_xprime_next[0][i] -= np.random.uniform(low=0., high=1e-3)
        
        log("x and x_prime: \n" + str(x_xprime_next))
        
        # 4. Run the duel and get y
        y_next = get_noisy_observation_dts(x_xprime_next, objective)
        log("y_next: \n" + str(y_next))
        
        # 5. Augment X and Y, and add symmetric points
        X = np.vstack([X, x_xprime_next, flip(x_xprime_next)])
        y = np.vstack([y, y_next, flip_y(y_next)])
        
        # Fit a GP with kernel k to Dj and learn pi(x).
        model = train_and_visualize(X, y, lengthscale=lengthscale, title="Run_{}_Evaluation_{}".format(run, evaluation))
        
        # Save model
        kernels_variance = []
        kernels_lengthscale = []
        for k in model.kernel.kernels:
            kernels_variance.append(k.variance.numpy())
            kernels_lengthscale.append(k.lengthscale.numpy())

        pickle.dump((X, y, 
                    tuple(kernels_variance),
                    tuple(kernels_lengthscale),
                    model.q_mu.numpy(),
                    model.q_sqrt.numpy()), 
                 open(results_dir + "Model_Run_{}_Evaluation_{}.p".format(run, evaluation), "wb"))
        
        # Get current best guess
        best_guess_results[run, evaluation] = best_guess(model, discrete_space, combs)

    X_results[run] = X
    y_results[run] = y


# In[ ]:


pickle.dump((X_results, y_results, best_guess_results), open(results_dir + "Xybestguess.p", "wb"))


# In[ ]:


global_min = np.min(objective(discrete_space))
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

