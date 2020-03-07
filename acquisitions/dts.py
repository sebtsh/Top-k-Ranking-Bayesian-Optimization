"""
Contains functions for the Dueling-Thompson sampling acquisition function by Gonzalez et al (2017).
"""

import numpy as np
import tensorflow as tf
import gpflow

from .. import fourier_features


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


def combinations(points):
    """
    Given d-dimensional points, return all pair combinations of those points
    :param points: tensor of shape (n, d)
    :return: tensor of shape (n ** 2, d * 2)
    """
    n = points.shape[0]
    d = points.shape[1]

    out = np.zeros((n*n, d*2))
    for i in range(n):
        for j in range(n):
            out[i*n+j][0:d] = points[i]
            out[i*n+j][d:d*2] = points[j]
    return out


def logistic(x):
    return 1 / (1 + tf.math.exp(-x))


def logistic_square(x):
    return tf.math.square(logistic(x))


def variance_integral(means, variances):
    return gpflow.quadrature.mvnquad(logistic_square, means, variances, H=50, Din=1)


def expected_integral(means, variances):
    return gpflow.quadrature.mvnquad(logistic, means, variances, H=50, Din=1)


def variance_logistic_f(m, x):
    means, variances = m.predict_f(x)
    variances = tf.expand_dims(variances, axis=1)
    expected_logistic_squared = tf.math.square(expected_integral(means, variances))
    return variance_integral(means, variances) - expected_logistic_squared


def sample_f(m, query_points, combs):
    """
    Generates a sample f using continuous Thompson sampling.
    :param m: gpflow model
    :param query_points: Input points corresponding to trained gpflow model. Tensor of shape (n, d)
    :param combs: tensor of shape (num_discrete_points ** 2, input_dims * 2). all combinations of discrete points
    :return: tensor of shape (num_discrete_points ** 2, 1)
    """

    X = np.expand_dims(combs, axis=0)

    phi, W, b = fourier_features.sample_fourier_features(X, m.kernel)
    phi_y = fourier_features.fourier_features(tf.expand_dims(query_points, axis=0), W, b)
    theta = fourier_features.sample_theta_variational(phi_y, m.q_mu, m.q_sqrt)
    return tf.squeeze(phi @ theta, axis=0)


def soft_copeland_maximizer(f_vals, discrete_space):
    """
    Given function evaluations, calculate the Condorcet winner.
    :param f_vals: tensor of shape (num_discrete_points ** 2, 1)
    :param discrete_space: tensor of shape (num_discrete_points, input_dims). all discrete points
    """
    num_discrete_points = int(np.sqrt(f_vals.shape[0]))
    soft_copeland = np.mean(np.reshape(logistic(f_vals),
                                       [num_discrete_points, num_discrete_points]), axis=1)  # (num_discrete_points)
    return discrete_space[np.argmax(soft_copeland)]
