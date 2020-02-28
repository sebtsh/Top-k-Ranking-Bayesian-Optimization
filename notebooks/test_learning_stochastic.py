import numpy as np
import gpflow
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import os
import pickle
import time 


from gpflow.utilities import set_trainable, print_summary

sys.path.append(os.path.split(os.path.split(os.getcwd())[0])[0]) # Move 2 levels up directory to import PBO
import PBO

SHOW_FIG = True

def forrester(x):
    """
    1-dimensional test function by Forrester et al. (2008)
    Defined as f(x) = (6x-2)^2 * sin(12x-4)
    :param x: float in domain [0, 1]
    """
    return (6*x-2)*(6*x-2)*np.sin(12*x-4)

def forrester_get_y(X):
    """
    Returns np array of shape (num_data, 1), indicating the input value with the most preferred Forrester value
    (lower is better)
    # param X: np array of shape (num_data, num_choices, 1)
    param X: [num_data] list of 2darray [:,d]
    """
    
    forr_minidx = [np.argmin(forrester(x), axis=0) for x in X]

    return [np.squeeze(np.take_along_axis(x, np.expand_dims(min_idx, axis=1), axis=0), axis=1)
                for x,min_idx in zip(X, forr_minidx)]


xx = np.linspace(0.0, 1.0, 100).reshape(100, 1)
plt.figure(figsize=(12, 6))
plt.plot(xx, forrester(xx), 'C0', linewidth=1)
plt.xlim(-0.0, 1.0)
if SHOW_FIG:
    plt.show()


def plot_gp(model, X, y, title):
    #Plotting code from GPflow authors

    ## generate test points for prediction
    xx = np.linspace(-0.1, 1.1, 100).reshape(100, 1)  # test points must be of shape (N, D)

    ## predict mean and variance of latent GP at test points
    mean, var = model.predict_f(xx)

    ## generate 10 samples from posterior
    samples = model.predict_f_samples(xx, 10)  # shape (10, 100, 1)

    ## plot 
    plt.figure(figsize=(12, 6))
    plt.plot(X, y, 'kx', mew=2)
    plt.plot(xx, mean, 'C0', lw=2)
    plt.fill_between(xx[:,0],
                     mean[:,0] - 1.96 * np.sqrt(var[:,0]),
                     mean[:,0] + 1.96 * np.sqrt(var[:,0]),
                     color='C0', alpha=0.2)

    plt.plot(xx, samples[:, :, 0].numpy().T, 'C0', linewidth=.5)
    plt.xlim(-0.1, 1.1)
    plt.title(title)
    
    if SHOW_FIG:
        plt.show()


num_train = 2000
indifference_threshold = 0.0
"""
if abs(f(x) - f(x')) < indifference_threshold
    user is indifferent (no preference) about x and x'
    observation in such as is represented as None
for example:
    X = [
        [[0.2], [0.4]],
        [[0.4], [0.7]],
        [[0.7], [0.95]]
        ]
    X = [np.array(x) for x in X]

    y = [np.array([[0.4]]), None, np.array([[0.7]])]

means

    0.2 is less preferred than 0.4
    0.4 and 0.7 are indifferent
    0.7 is more preferred than 0.95
"""

# # Sample data
# X = [
#     [[0.2], [0.4], [0.9]],
#     [[0.4], [0.7], [0.23], [0.9]],
#     [[0.2], [0.23]]
#     ]
# X = [[[0.2], [0.4]],
#     [[0.4], [0.7]],
#     [[0.76], [0.9]],
#     [[0.9], [0.2]],
#     [[0.7], [0.9]],
#     [[0.2], [0.7]],
#     [[0.7], [0.76]]]

# X = [np.array(x) for x in X]

# y = forrester_get_y(X)


# # X = [[[0.2], [0.4]],
# #     [[0.4], [0.7]],
# #     [[0.7], [0.95]]]

# # X = [np.array(x) for x in X]

# # y = [np.array([[0.4]]), None, np.array([[0.7]])]
# # # y = [np.array([[0.4]]), np.array([[0.4]]), np.array([[0.7]])]

# idx_to_val_dict, val_to_idx_dict = PBO.models.learning_stochastic.populate_dicts(X)
# D_idxs, max_idxs = PBO.models.learning_stochastic.val_to_idx(X, y, val_to_idx_dict)


# print("X")
# print(X)
# print("y")
# print(y)

# print("idx_to_val_dict")
# print(idx_to_val_dict)
# print(val_to_idx_dict)

# print("D_idxs")
# print(D_idxs.read(0))
# print(D_idxs.read(1))
# print("max_idxs")
# print(max_idxs)


# q_mu, q_sqrt, u, inputs, k = PBO.models.learning_stochastic.train_model_fullcov(
#                                     X, y, 
#                                     num_inducing=5, 
#                                     num_steps=num_train, 
#                                     indifference_threshold=indifference_threshold)


# likelihood = gpflow.likelihoods.Gaussian()
# model = PBO.models.learning.init_SVGP_fullcov(q_mu, q_sqrt, u, k, likelihood)

# u_mean = q_mu.numpy()
# inducing_vars = u.numpy()

# plot_gp(model, inducing_vars, u_mean, "GP")



num_runs = 2
num_evals = 2
num_samples = 20
num_choices = 2
input_dims = 1
num_maximizers = 10
num_init_points = 2
num_inducing_init = 5
num_discrete_points = 10000


def train_and_visualize(X, y, num_inducing, title, indifference_threshold=0.0):
    
    # Train model with data
    q_mu, q_sqrt, u, inputs, k = PBO.models.learning_stochastic.train_model_fullcov(
                                                X, y, 
                                                num_inducing=num_inducing,
                                                num_steps=num_train,
                                                indifference_threshold=indifference_threshold)

    likelihood = gpflow.likelihoods.Gaussian()
    model = PBO.models.learning.init_SVGP_fullcov(q_mu, q_sqrt, u, k, likelihood)
    u_mean = q_mu.numpy()
    inducing_vars = u.numpy()
    
    print("Kernel length-scale: {}".format(k.lengthscale.value))

    # Visualize model
    plot_gp(model, inducing_vars, u_mean, title)
    
    return model, inputs, u_mean, inducing_vars



def best_guess(model):
    """
    Returns a GP model's best guess of the global maximum of f.
    """
    xx = np.linspace(0.0, 1.0, num_discrete_points).reshape(num_discrete_points, 1)
    res = model.predict_f(xx)[0].numpy()
    return xx[np.argmax(res)]




num_data_at_end = int((num_init_points-1) * num_init_points / 2 + num_evals)
X_results = np.zeros([num_runs, num_data_at_end, num_choices, input_dims])
y_results = np.zeros([num_runs, num_data_at_end, 1])
best_guess_results = np.zeros([num_runs, num_evals, input_dims])


np.random.seed(0)
init_points = np.random.uniform(size=[num_runs, num_init_points, input_dims])
num_combs = int((num_init_points-1) * num_init_points / 2)
init_vals = np.zeros([num_runs, num_combs, num_choices, input_dims])
for run in range(num_runs):
    cur_idx = 0
    for init_point in range(num_init_points-1):
        for next_point in range(init_point+1, num_init_points):
            init_vals[run, cur_idx, 0] = init_points[run, init_point]
            init_vals[run, cur_idx, 1] = init_points[run, next_point]
            cur_idx += 1





for run in range(num_runs):
    print("Beginning run %s" % (run))
    
    X = init_vals[run]
    y = forrester_get_y(X)
    
    model, inputs, u_mean, inducing_vars = train_and_visualize(
                                    X, y, 
                                    num_inducing_init, 
                                    "Run {}: Initial model".format(run),
                                    indifference_threshold)

    for evaluation in range(num_evals):
        print("Beginning evaluation %s" % (evaluation)) 

        # Sample possible next queries
        samples = PBO.acquisitions.pes.sample_inputs(inputs, num_samples, num_choices)
        # (num_samples*num_inputs, num_choices, input_dims)

        # Sample maximizers
        print("Evaluation %s: Sampling maximizers" % (evaluation))
        maximizers = PBO.fourier_features.sample_maximizers(X=inducing_vars,
                                                            y=u_mean,
                                                            count=num_maximizers,
                                                            D=100,
                                                            model=model)
        print("Maximizer samples: {}".format(maximizers))

        # Calculate PES value I for each possible next query
        print("Evaluation %s: Calculating I" % (evaluation))
        I_vals = PBO.acquisitions.pes.I_batch(samples, maximizers, model)
        # (num_samples * num_inputs)

        I_vals1, log_p_xstar, log_p_obs, log_p_xstar_obs\
                 = PBO.acquisitions.rank_pes.I_batch(
                                samples.numpy(), 
                                maximizers.numpy(), 
                                model, 
                                num_samples=1000, 
                                indifference_threshold = indifference_threshold)

        # (num_samples * num_inputs)
        print("I_vals: {}".format(I_vals))
        print("I_vals1: {}".format(I_vals1))
        print("Difference in I_vals RMS: {}".format( np.sqrt(np.mean(np.square(I_vals - I_vals1)))) )
        raise Exception("test rank_pes")


        # Select query that maximizes I
        next_idx = np.argmax(I_vals)
        next_query = samples[next_idx]
        print("Evaluation %s: Next query is %s with I value of %s" % (evaluation, next_query, I_vals[next_idx]))

        X = np.concatenate([X, [next_query]])
        # Evaluate objective function
        y = forrester_get_y(X)
        
        print("Evaluation %s: Training model" % (evaluation))
        model, inputs, u_mean, inducing_vars = train_and_visualize(X, y, 
                                                                   num_inducing_init + evaluation + 1, 
                                                                   "Run {}: Evaluation {}".format(run, evaluation))

        best_guess_results[run, evaluation, :] = best_guess(model)

    X_results[run] = X
    y_results[run] = y


