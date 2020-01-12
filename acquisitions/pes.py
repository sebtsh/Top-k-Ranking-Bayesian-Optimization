"""
Contains functions for the Predictive Entropy Search acquisition function.
Formulation by Nguyen Quoc Phong.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def sample_maximizers_simple(model, count, num_discrete_points):
    """
    Samples maximizers from the GP by sampling real values at each discrete point and finding the argmax
    :param model: gpflow model
    :param count: number of maximizers
    :param num_discrete_points: number of discrete points over input space to sample
    :return: tensor of shape (count, 1)
    """

    xx = np.linspace(0.0, 1.0, num_discrete_points).reshape(num_discrete_points, 1)
    samples = model.predict_f_samples(xx, count) # (count, num_discrete_points)
    samples_argmax = np.squeeze(np.argmax(samples, axis=1), axis=1)
    return np.expand_dims(np.take(xx, samples_argmax), axis=1)


def fourier_features(X, kernel, D=100):
    """
    Fourier feature mapping of X for the RBF kernel
    :param X: tensor of shape (count, n, d)
    :return: tensor of shape (count, n, D)
    """
    count = X.shape[0]
    n = X.shape[1]
    d = X.shape[2]

    W = tf.random.normal(shape=(count, D, d),
                         mean=0.0,
                         stddev=1.0 / tf.sqrt(kernel.lengthscale),
                         dtype=tf.float64)
    b = tf.random.uniform(shape=(count, D, 1),
                          minval=0,
                          maxval=2 * np.pi,
                          dtype=tf.float64)
    WX_b = W @ tf.linalg.matrix_transpose(X) + b  # (count, D, n)
    alpha = kernel.variance ** d

    return tf.sqrt(2.0 * alpha / D) * tf.cos(tf.linalg.matrix_transpose(WX_b)), W, b  # (count, n, D)


def sample_maximizers(X, y, count, D, model, num_steps=3000):
    """
    Samples from the posterior over the global maximizer using the method by Shah & Ghahramani (2015). Approximates
    the RBF kernel with its Fourier dual. Samples random Fourier features, constructs a linear model and computes
    the argmax using gradient-based optimization.

    :param X: input points, tensor with shape (n, d)
    :param y: objective function evaluations, tensor with shape (n, 1)
    :param count: number of maximizers to sample
    :param D: number of Fourier features to use
    :param model: gpflow model that uses the RBF kernel and has been optimized
    :param num_steps: int that specifies how many optimization steps to take
    """
    d = X.shape[1]

    X = tf.tile(tf.expand_dims(X, axis=0), [count, 1, 1])  # (count, n, d)

    # Sample random features phi and get W and b
    phi, W, b = fourier_features(X, model.kernel, D)  # phi has shape (count, n, D)

    # Sample posterior weights theta
    A = (tf.linalg.matrix_transpose(phi) @ phi) + \
        tf.expand_dims(model.likelihood.variance * tf.eye(D, dtype=tf.float64), axis=0)
    A_inv = tf.linalg.inv(A)  # (count, D, D)
    theta_mean = tf.squeeze(A_inv @ tf.linalg.matrix_transpose(phi) @ tf.expand_dims(y, axis=0))  # (count, D)
    theta_var = model.likelihood.variance * A_inv
    theta_dist = tfp.distributions.MultivariateNormalFullCovariance(loc=theta_mean,
                                                                    covariance_matrix=theta_var)
    theta = tf.expand_dims(tf.dtypes.cast(theta_dist.sample(), dtype=tf.float64), axis=2)  # (count, D, 1)

    def construct_maximizer_objective(x_star_latent):
        # Construct g
        x_star = tf.sigmoid(x_star_latent)  # Constrains the value of x_star to between 0 and 1
        WX_b = W @ tf.linalg.matrix_transpose(x_star) + b  # (count, D, 1)
        alpha = model.kernel.variance ** d
        g = tf.reduce_sum((tf.sqrt(2.0 * alpha / D) * tf.cos(tf.linalg.matrix_transpose(WX_b))) @ theta)
        return -g

    # Compute x_star using gradient based methods
    optimizer = tf.keras.optimizers.Adam()
    x_star_latent = tf.Variable(tf.random.normal(shape=(count, 1, d), dtype=tf.dtypes.float64))
    loss = lambda: construct_maximizer_objective(x_star_latent)

    for i in range(num_steps):
        optimizer.minimize(loss, var_list=[x_star_latent])
        if i % 200 == 0:
            print('Loss at step %s: %s' % (i, loss().numpy()))

    return tf.squeeze(tf.sigmoid(x_star_latent), axis=1)


@tf.function
def samples_likelihood(z, f_z):
    """
    :param z: tuple of the form (x, chi) where chi is a tensor of shape (num_choices, d) and x is an index to
    the most preferred input in chi
    :param f_z: tensor of shape (num_samples, num_choices) with f values corresponding to chi
    :return: tensor of shape (num_samples)
    """
    x = z[0]
    return tf.exp(f_z[:, x]) / tf.reduce_sum(tf.exp(f_z), axis=1)


@tf.function
def log_p_x_star_cond_D(x_star, model, num_samples=1000):
    """
    Among possible maximizers of the function, calculates the log probability that each is the global maximizer.
    Samples from the model and calculates an expectation from the samples.
    Returns a tensor of shape (num_max)
    :param x_star: possible maximizers, tensor of shape (num_max, d)
    :param model: GPflow model
    """
    num_max = x_star.shape[0]

    samples = tf.squeeze(model.predict_f_samples(x_star, num_samples))  # (num_samples, num_max)
    samples_argmax = tf.argmax(samples, 1, output_type=tf.int32)  # (num_samples)
    count = tf.zeros(num_max, dtype=tf.float64) + tf.keras.backend.epsilon()  # Add epsilon to avoid dividing by 0
    count += tf.math.bincount(samples_argmax, minlength=num_max, dtype=tf.float64)
    return tf.math.log(count / (num_samples + num_max * tf.keras.backend.epsilon()))


@tf.function
def log_p_z_cond_D_x_star(z, x_star, log_p_x_star_cond, model, num_samples=1000):
    """
    Returns a tensor of shape (num_max)
    :param z: tuple of the form (x, chi) where chi is a tensor of shape (num_choices, d) and x is an index to
    the most preferred input in chi
    :param x_star: possible maximizers, tensor of shape (num_max, d)
    :param log_p_x_star_cond_D: tensor of shape (num_max)
    :param model: GPflow model
    """
    num_max = x_star.shape[0]

    x_vals = tf.concat([x_star, z[1]], 0)  # f(x_star), f(z)
    samples = tf.squeeze(model.predict_f_samples(x_vals, num_samples))  # (num_samples, num_max + num_choices)
    samples_x_star = samples[:, :num_max]  # (num_samples, num_max)
    samples_x_star_argmax = tf.argmax(samples_x_star, 1, output_type=tf.int32)  # (num_samples)

    p_z_cond_f_z = tf.scatter_nd(indices=tf.expand_dims(samples_x_star_argmax, axis=1),
                                 updates=samples_likelihood(z, samples[:, num_max:]),
                                 shape=tf.constant([num_max])) + tf.keras.backend.epsilon()

    return tf.math.log(p_z_cond_f_z / num_samples) - log_p_x_star_cond


def I(chi, x_star, model):
    """
    Predictive Entropy Search acquisition function.
    :param chi: input points in a query, tensor of shape (num_choices, d)
    :param x_star: possible maximizers, tensor of shape (num_max, d)
    :param model: GPflow model
    """

    num_choices = chi.shape[0]
    num_max = x_star.shape[0]

    log_p_x_star_cond = log_p_x_star_cond_D(x_star, model)  # (num_max)

    expected_log = tf.Variable(initial_value=tf.zeros((num_max, num_choices), dtype=tf.float64),
                               dtype=tf.float64,
                               shape=(num_max, num_choices))  # p(z|D,x_star)*log((p|D,x_star)/p(z|D))
    for i in range(num_choices):
        z = (i, chi)
        log_p_z_cond = log_p_z_cond_D_x_star(z, x_star, log_p_x_star_cond, model)
        norm_constant = tf.reduce_logsumexp(log_p_x_star_cond + log_p_z_cond)
        expected_log[:, i].assign(tf.exp(log_p_z_cond) * (log_p_z_cond - norm_constant))

    sum_over_preferred = tf.reduce_sum(expected_log, axis=1)

    return tf.reduce_sum(tf.exp(log_p_x_star_cond) * sum_over_preferred)


def I_batch(chi_batch, x_star, model):
    """
    :param chi_batch: input points in a query, tensor of shape (num_queries, num_choices, d)
    :param x_star: possible maximizers, tensor of shape (num_max, d)
    :param model: GPflow model
    """
    return np.array([I(chi, x_star, model) for chi in chi_batch])


def sample_inputs(current_inputs, num_samples, num_choices, min_val=0.0, max_val=1.0):
    """
    Uniformly samples random inputs to query objective function. Sampled inputs must have
    existing data points among the choices, otherwise the learned function values for the
    input choices will be independent of the already learned function values for other data points.
    Returns np array of shape (num_samples*num_inputs, num_choices, input_dims)
    :param current_inputs: np array of shape (num_inputs, input_dims)
    :param num_samples: int, number of random values to permutate with existing inputs
    :param num_choices: int, number of choices in an input query
    :param min_val: float, minimum value of sampled random values
    :param max_val: float, max value of sampled random values
    """
    num_inputs = current_inputs.shape[0]
    input_dims = current_inputs.shape[1]
    samples = np.zeros([num_samples * num_inputs, num_choices, input_dims])
    uniform_samples = np.random.uniform(low=min_val,
                                        high=max_val,
                                        size=(num_samples, num_choices - 1, input_dims))

    for i in range(num_inputs):
        for j in range(num_samples):
            cur_idx = i * num_samples + j
            samples[cur_idx, 0, :] = current_inputs[i]
            samples[cur_idx, 1:, :] = uniform_samples[j]

    return tf.constant(samples)