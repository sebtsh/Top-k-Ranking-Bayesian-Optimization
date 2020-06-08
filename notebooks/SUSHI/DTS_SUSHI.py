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


def log(message):
    print(str(datetime.datetime.now()) + ': ' + message)


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
acquisition_name = "DTS"
experiment_name = "PBO" + "_" + acquisition_name + "_" + objective_name


# In[ ]:


num_runs = 10
num_evals = 35
num_choices = 2
input_dims = 6
num_init_prefs = 10
num_fourier_features = 200
num_in_subset = 100


# In[ ]:


regularizer_lengthscale_mean_over_range = 0.2
regularizer_lengthscale_std_over_range = 0.5
input_range = objective_high - objective_low
lengthscale_mean_regularizer = input_range * regularizer_lengthscale_mean_over_range
lengthscale_std_regularizer = input_range * regularizer_lengthscale_std_over_range
lengthscale = lengthscale_mean_regularizer


# In[ ]:


results_dir = os.getcwd() + '/results/' + experiment_name + '/'

try:
    # Create target Directory
    os.makedirs(results_dir)
    print("Directory " , results_dir ,  " created ") 
except FileExistsError:
    print("Directory " , results_dir ,  " already exists")


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
        
    return m


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


# Generate rank dictionary and immediate regret dictionary.

# In[ ]:


fval_idx_tuples = pickle.load(open("fval_idx_tuples.p", "rb"))


# In[ ]:


rank_dict = {}

for i in range(len(fval_idx_tuples)):
    rank_dict[features[fval_idx_tuples[i][1]].data.tobytes()] = i + 1


# This function is our main metric for the performance of the acquisition function.

# In[ ]:


def get_max_sushi(m, features, combs, rank_dict):
    """
    Specifically for DTS
    :param model: gpflow model
    :param features: sushi features
    :param rank_dict: dictionary from sushi idx to place in ranking
    :return: tuple (index of max sushi, rank)
    """
    y_vals = m.predict_y(combs)[0]
    num_discrete_points = int(np.sqrt(y_vals.shape[0]))
    soft_copeland = np.mean(np.reshape(y_vals,
                                       [num_discrete_points, num_discrete_points]), axis=1)  # (num_discrete_points)
    max_idx = np.argmax(soft_copeland)
    
    return (max_idx, rank_dict[features[max_idx].data.tobytes()])


# Create the initial values for each run:

# In[ ]:


np.random.seed(0)
random_indices = np.zeros([num_runs, num_init_prefs, num_choices], dtype=np.int64)
for i in range(num_runs):
    random_indices[i] = np.random.choice(features.shape[0], [num_init_prefs, num_choices], replace=False)


# In[ ]:


init_vals = np.take(features, random_indices, axis=0)
init_vals = np.reshape(init_vals, (num_runs, num_init_prefs, num_choices * input_dims))


# Store the results in these arrays:

# In[ ]:


num_data_at_end = (num_init_prefs + num_evals) * 2
X_results = np.zeros([num_runs, num_data_at_end, input_dims * num_choices])
y_results = np.zeros([num_runs, num_data_at_end, 1])
immediate_regret = np.zeros([num_runs, num_evals], np.int32)


# In[ ]:


def array_in(a, b):
    """
    a: 1-D array with shape (d, )
    b: 2-D array with shape (n, d)
    :return: bool
    """
    for i in range(b.shape[0]):
        if np.allclose(a, b[i]):
            return True
    return False


# In[ ]:


for run in range(num_runs):
    log("Starting run {}".format(run))
    #Fit a GP with kernel k to Dn
    
    X = init_vals[run]
    y = get_noisy_observation_dts(X, objective)
    
    X = np.vstack([X, flip(X)])
    y = np.vstack([y, flip_y(y)])
    
    model = train_and_visualize(X, y, lengthscale=lengthscale, title="Run_{}_Initial_model".format(run))
    
    for evaluation in range(num_evals):
        log("Starting evaluation " + str(evaluation))

        is_valid_query = False
        num_tries = 0
        while not is_valid_query:
            discrete_space = features
            combs = PBO.acquisitions.dts.combinations(features)
            
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

            # If both x and x' are equal, or the query has already been made, will cause Fourier features matrix
            # to become non-invertible later on
            if np.all(np.equal(x_xprime_next, flip(x_xprime_next))) or array_in(x_xprime_next, X):
                log("Invalid query, resampling f")
                print("X:")
                print(X)
                print("Attempted query:")
                print(x_xprime_next)
                num_tries += 1
                if num_tries >= 10:
                    raise ValueError
            else:
                log("x and x_prime: \n" + str(x_xprime_next))
                is_valid_query = True
        
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
        
        (max_idx, rank) = get_max_sushi(model, features, combs, rank_dict)
        immediate_regret[run, evaluation] = rank - 1
        
        print("Maximizing sushi has index {} and rank {}".format(max_idx, rank)) 

    X_results[run] = X
    y_results[run] = y
    print("Run {} immediate regret: ".format(run))
    print(immediate_regret[run])


# In[ ]:


pickle.dump((X_results, y_results, immediate_regret), open(results_dir + "res.p", "wb"))

