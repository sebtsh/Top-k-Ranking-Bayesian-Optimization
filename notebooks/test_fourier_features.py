import numpy as np
import gpflow
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import os
import pickle
import time 

physical_devices = tf.config.list_physical_devices('GPU') 
try: 
  tf.config.experimental.set_memory_growth(physical_devices[0], True) 
except: 
  # Invalid device or cannot modify virtual devices once initialized. 
  pass

from gpflow.utilities import set_trainable, print_summary

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


# xx = np.linspace(0.0, 1.0, 100).reshape(100, 1)
# plt.figure(figsize=(12, 6))
# plt.plot(xx, forrester(xx), 'C0', linewidth=1)
# plt.xlim(-0.0, 1.0)
# if SHOW_FIG:
#     plt.show()


def plot_gp(model, X, y, title):
    #Plotting code from GPflow authors

    ## generate test points for prediction
    xx = np.linspace(-0.1, 1.1, 100).reshape(100, 1)  # test points must be of shape (N, D)

    ## predict mean and variance of latent GP at test points
    mean, var = model.predict_f(xx)

    ## generate 10 samples from posterior
    samples = model.predict_f_samples(xx, 10)  # shape (10, 100, 1)

    fig, ax = plt.subplots()

    ## plot 
    ax.plot(X, y, 'kx', mew=2)
    ax.plot(xx, mean, 'C0', lw=2)
    ax.fill_between(xx[:,0],
                     mean[:,0] - 1.96 * np.sqrt(var[:,0]),
                     mean[:,0] + 1.96 * np.sqrt(var[:,0]),
                     color='C0', alpha=0.2)

    ax.plot(xx, samples[:, :, 0].numpy().T, 'C0', linewidth=.5)
    ax.set_xlim(-0.1, 1.1)
    ax.set_title(title)


k = gpflow.kernels.RBF(lengthscale=0.1)

u = tf.constant(np.array([0., 0.25, 0.5, 0.9]).reshape(-1,1))
q_mu = tf.constant(np.array([0.5, -3., 0.5, -8.5]).reshape(-1,1))
q_sqrt = tf.constant(np.eye(4).reshape(1,4,4))

likelihood = gpflow.likelihoods.Gaussian()
model = PBO.models.learning.init_SVGP_fullcov(q_mu, q_sqrt, u, k, likelihood)

u_mean = q_mu.numpy()
inducing_vars = u.numpy()

plot_gp(model, inducing_vars, u_mean, "GP from GPflow")

count = 100

X_np = np.linspace(-0.1, 1.1, 100).reshape(100, 1)  # test points must be of shape (N, D)
X = tf.expand_dims(X_np, axis=0)

u_features, W, b = PBO.fourier_features.sample_fourier_features(tf.tile(tf.reshape(u, (1,-1,1)), [count, 1, 1]), k, D=100)
# (count,nu,D)


# theta = PBO.fourier_features.sample_theta_variational(u_features, q_mu, q_sqrt, model.likelihood.variance)
# # (count,nu,D)
theta = PBO.fourier_features.sample_theta_variational(u_features, q_mu, q_sqrt, 0)
# (count,nu,D)

ys = PBO.fourier_features.fourier_features(X, W, b, k) @ theta 
# (5, 100, 1)
ys_np = ys.numpy()


fig, ax = plt.subplots()
for i in range(count):
  ax.plot(X_np.squeeze(), ys_np[i,...].squeeze())
ax.set_title("Samples from GP using fourier features")


maximizers = PBO.fourier_features.sample_maximizers(u, count, n_init=50, D=100, model=model, min_val=-0.1, max_val=1.1, num_steps=500)
# (count,xdim)

maximizers = maximizers.numpy()

fig, ax = plt.subplots()
ax.hist(maximizers.squeeze())
ax.set_xlim(-0.1, 1.1)
ax.set_title("Histogram of sampled maximizers")

plt.show()
