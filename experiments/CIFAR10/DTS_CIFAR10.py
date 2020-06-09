#!/usr/bin/env python
# coding: utf-8

# # Preferential Bayesian Optimization: Dueling-Thompson Sampling
# 
# Implementation of the algorithm by Gonzalez et al (2017).
# 
# Over the CIFAR-10 dataset, we define an arbitrary preference as such (with class number in parentheses):
# 
# Airplane (0) > Automobile (1) > Ship (8) > Truck (9) > Bird (2) > Cat (3) > Deer (4) > Dog (5) > Frog (6) > Horse (7)

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


cifar_embedding = pickle.load( open( "cifar_embedding_reduced.p", "rb" ) )


# In[ ]:


embedding_to_class = pickle.load( open( "embedding_to_class_reduced.p", "rb" ) )


# In[ ]:


objective = lambda x: PBO.objectives.cifar(x, embedding_to_class)
objective_low = np.min(cifar_embedding)
objective_high = np.max(cifar_embedding)
objective_name = "CIFAR"
acquisition_name = "DTS"
experiment_name = acquisition_name + "_" + objective_name


# In[ ]:


def get_class(x):
    """
    :param x: tensor of shape (..., 2). CIFAR-10 embeddings
    :return: tensor of shape (..., 1). last dim is int from 0-9 representing class
    """
    shape = x.shape[:-1]
    raveled = np.reshape(x, [-1, 2])
    raveled_shape = raveled.shape[:-1]
    raveled_classes = np.zeros((raveled_shape[0], 1), dtype=np.int8)
    
    for i in range(raveled_shape[0]):
        raveled_classes[i] = embedding_to_class[raveled[i].data.tobytes()]
        
    return np.reshape(raveled_classes, shape + (1,))


# In[ ]:


regularizer_lengthscale_mean_over_range = 0.2
regularizer_lengthscale_std_over_range = 0.5
input_range = objective_high - objective_low
lengthscale_mean_regularizer = input_range * regularizer_lengthscale_mean_over_range
lengthscale_std_regularizer = input_range * regularizer_lengthscale_std_over_range
lengthscale = lengthscale_mean_regularizer


# In[ ]:


num_runs = 10
num_evals = 35
num_choices = 2
input_dims = 2
objective_dim = input_dims
num_init_prefs = 6
num_fourier_features = 200
num_in_subset = 100


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


def pref_inversions(model):
    """
    Method to evaluate models over discrete preference rankings. Given an objective preference ranking over classes, 
    we calculate the average mean the model assigns to each class, sort the classes according to this average mean,
    then calculate the number of inversions required to reach the desired objective preference ranking. 0 inversions
    means the model has learned the preference ranking perfectly. The more inversions, the further away the model is.
    """
    def count_inversions(input_list):
        def swap(lst, i, j):
            tmp = lst[j]
            lst[j] = lst[i]
            lst[i] = tmp

        lst = input_list.copy()
        num_inversions = 0
        changed = True
        while changed:
            changed = False
            for i in range(len(lst) - 1):
                if lst[i] > lst[i+1]:
                    swap(lst, i, i+1)
                    num_inversions += 1
                    changed = True
                    
        return num_inversions
    
    
    class_to_posval = {0: -0.1,
                     1: -0.2,
                     8: -0.3,
                     9: -0.4,
                     2: -0.5,
                     3: -0.6,
                     4: -0.7,
                     5: -0.8,
                     6: -0.9,
                     7: -1.}  # higher is more preferred here
    
    fvals = model.predict_y(cifar_embedding)[0]
    indices = get_class(cifar_embedding)
    
    average_f = tf.scatter_nd(indices=indices,
                   updates=np.squeeze(fvals),
                   shape=tf.constant([10]))/5000
    sorted_f = sorted(list(zip(average_f, range(10))))
    
    model_posvals = []
    for pair in sorted_f:
        model_posvals.append(class_to_posval[pair[1]])
        
    return count_inversions(model_posvals)


# In[ ]:


def get_maximizing_class(model):
    fvals = model.predict_y(cifar_embedding)[0]
    indices = get_class(cifar_embedding)
    
    average_f = tf.scatter_nd(indices=indices,
                   updates=np.squeeze(fvals),
                   shape=tf.constant([10]))/5000
    sorted_f = sorted(list(zip(average_f, range(10))))
    return sorted_f[9][1]


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
random_indices = np.random.choice(cifar_embedding.shape[0], [num_runs, num_init_prefs, num_choices], replace=False)
init_vals = np.take(cifar_embedding, random_indices, axis=0)
init_vals = np.reshape(init_vals, (num_runs, num_init_prefs, num_choices * input_dims))


# Store the results in these arrays:

# In[ ]:


num_data_at_end = (num_init_prefs + num_evals) * 2
X_results = np.zeros([num_runs, num_data_at_end, input_dims * num_choices])
y_results = np.zeros([num_runs, num_data_at_end, 1])
inversion_results = np.zeros([num_runs, num_evals], np.int32)
max_class_results = np.zeros([num_runs, num_evals], np.int32)


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
            # Get random subset of cifar_embedding to use per evaluation
            subset_indices = np.random.choice(cifar_embedding.shape[0], num_in_subset, replace=False)
            discrete_space = np.take(cifar_embedding, subset_indices, axis=0)
            combs = PBO.acquisitions.dts.combinations(discrete_space)
            
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
            if np.all(np.equal(x_xprime_next, flip(x_xprime_next))) or x_xprime_next in X:
                log("Invalid query, resampling f")
                num_tries += 1
                if num_tries >= 100:
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
        
        inversion_results[run, evaluation] = pref_inversions(model)
        max_class_results[run, evaluation] = get_maximizing_class(model)
        
        print("Inversions: {}, maximizing class: {}".format(inversion_results[run, evaluation], 
                                                           max_class_results[run, evaluation]))

    X_results[run] = X
    y_results[run] = y


# In[ ]:


pickle.dump((X_results, y_results, inversion_results, max_class_results), open(results_dir + "Xybestguess.p", "wb"))


# In[ ]:


class_to_ir = {0:0, 1:1, 8:2, 9:3, 2:4, 3:5, 4:6, 5:7, 6:8, 7:9}


# In[ ]:


ir = np.zeros(max_class_results.shape)
for i in range(num_runs):
    for j in range(num_evals):
        ir[i, j] = max_class_results[i, j]
        
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


# In[ ]:




