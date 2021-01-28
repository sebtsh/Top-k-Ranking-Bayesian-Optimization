#!/usr/bin/env python
# coding: utf-8

# # Preferential Bayesian Optimization: Multinomial Predictive Entropy Search
# This notebook demonstrates the use of the Multinomial Predictive Entropy Search (PES) acquisition function on ordinal (preference) data.
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


cifar_embedding = pickle.load( open( "cifar_embedding_reduced.p", "rb" ) )


# In[ ]:


embedding_to_class = pickle.load( open( "embedding_to_class_reduced.p", "rb" ) )


# In[ ]:


objective = lambda x: PBO.objectives.cifar(x, embedding_to_class)
objective_low = np.min(cifar_embedding)
objective_high = np.max(cifar_embedding)
objective_name = "CIFAR"
acquisition_name = "MPES"
experiment_name = acquisition_name + "_" + objective_name


# In[ ]:


num_runs = 10
num_evals = 35
num_samples = 100
num_choices = 2
input_dims = 2
objective_dim = input_dims # CHANGE 1: require the objective dim
num_maximizers = 20
num_maximizers_init = 50
num_fourier_features = 1000
num_init_prefs = 6 # CHANGE 2: randomly initialize with some preferences
num_discrete_per_dim = 1000  # for plotting


# In[ ]:


results_dir = os.getcwd() + '/results/' + experiment_name + '/'

try:
    # Create target Directory
    os.makedirs(results_dir)
    print("Directory " , results_dir ,  " created ") 
except FileExistsError:
    print("Directory " , results_dir ,  " already exists")


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


# This function is our main metric for the performance of the acquisition function.

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
    
    fvals = model.predict_f(cifar_embedding)[0]
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
    fvals = model.predict_f(cifar_embedding)[0]
    indices = get_class(cifar_embedding)
    
    average_f = tf.scatter_nd(indices=indices,
                   updates=np.squeeze(fvals),
                   shape=tf.constant([10]))/5000
    sorted_f = sorted(list(zip(average_f, range(10))))
    return sorted_f[9][1]


# Store the results in these arrays:

# In[ ]:


num_data_at_end = int(num_init_prefs + num_evals)
X_results = np.zeros([num_runs, num_data_at_end, num_choices, input_dims])
y_results = np.zeros([num_runs, num_data_at_end, 1, input_dims])
inversion_results = np.zeros([num_runs, num_evals], np.int32)
max_class_results = np.zeros([num_runs, num_evals], np.int32)


# Create the initial values for each run:

# In[ ]:


np.random.seed(0)
random_indices = np.random.choice(cifar_embedding.shape[0], [num_runs, num_init_prefs, num_choices])
init_vals = np.take(cifar_embedding, random_indices, axis=0)


# The following loops carry out the Bayesian optimization algorithm over a number of runs, with a fixed number of evaluations per run.

# In[ ]:


for run in range(num_runs):  # CHECK IF STARTING RUN IS CORRECT
    print("Beginning run %s" % (run))
    
    X = init_vals[run]
    y = get_noisy_observation(X, objective)
    
    model, inputs, u_mean, inducing_vars = train_and_visualize(X, y, "Run_{}:_Initial_model".format(run))

    for evaluation in range(num_evals):
        print("Beginning evaluation %s" % (evaluation)) 
        
        success = False
        fail_count = 0
        while not success:
            # Sample possible next queries
            samples = PBO.acquisitions.pes.sample_inputs_discrete(current_inputs=inputs,
                                                                data=cifar_embedding,
                                                                num_samples=num_samples,
                                                                num_choices=num_choices)

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

            if fail_count >= 10:
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
                     model.q_sqrt, 
                     maximizers), 
                    open(results_dir + "Model_Run_{}_Evaluation_{}.p".format(run, evaluation), "wb"))

        inversion_results[run, evaluation] = pref_inversions(model)
        max_class_results[run, evaluation] = get_maximizing_class(model)
        
        print("Inversions: {}, maximizing class: {}".format(inversion_results[run, evaluation], 
                                                           max_class_results[run, evaluation]))

    X_results[run] = X
    y_results[run] = y


# In[ ]:


pickle.dump((X_results, y_results, inversion_results, max_class_results), open(results_dir + "PES_CIFAR_runs2-10.p", "wb"))


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

