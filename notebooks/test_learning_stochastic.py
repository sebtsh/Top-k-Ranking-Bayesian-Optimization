import numpy as np
import gpflow
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import os
import pickle

from gpflow.utilities import set_trainable, print_summary
gpflow.config.set_default_summary_fmt("notebook")

sys.path.append(os.path.split(os.path.split(os.getcwd())[0])[0]) # Move 2 levels up directory to import PBO
import PBO



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
    
    plt.show()


# Sample data
X = [
    [[0.2], [0.4], [0.9]],
    [[0.4], [0.7], [0.23], [0.9]],
    [[0.2], [0.23]]
    ]
X = [np.array(x) for x in X]

y = forrester_get_y(X)



idx_to_val_dict, val_to_idx_dict = PBO.models.learning_stochastic.populate_dicts(X)
D_idxs, max_idxs = PBO.models.learning_stochastic.val_to_idx(X, y, val_to_idx_dict)


print("X")
print(X)
print("y")
print(y)

print("idx_to_val_dict")
print(idx_to_val_dict)
print(val_to_idx_dict)

print("D_idxs")
print(D_idxs.read(0))
print(D_idxs.read(1))
print("max_idxs")
print(max_idxs)


q_mu, q_sqrt, u, inputs, k = PBO.models.learning_stochastic.train_model_fullcov(X, y, num_inducing=5, num_steps=2000)


likelihood = gpflow.likelihoods.Gaussian()
model = PBO.models.learning.init_SVGP_fullcov(q_mu, q_sqrt, u, k, likelihood)

u_mean = q_mu.numpy()
inducing_vars = u.numpy()

plot_gp(model, inducing_vars, u_mean, "GP")

