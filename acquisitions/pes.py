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


def sample_maximizers(X, y, count, D, variance, num_steps=5000):
    """
    Samples from the posterior over the global maximizer using the method by Shah & Ghahramani (2015). Approximates
    the RBF kernel with its Fourier dual. Samples random Fourier features, constructs a linear model and computes
    the argmax using gradient-based optimization.

    :param X: input points, tensor with shape (n, d)
    :param y: objective function evaluations, tensor with shape (n, 1)
    :param count: number of maximizers to sample
    :param D: number of Fourier features to use
    :param variance: #TODO: Set as same as kernel variance from SVGP?
    :param num_steps: int that specifies how many optimization steps to take
    """

    d = X.shape[1]

    def rff_z(X, W):
        """
        From RandomFourierFeatures by Tetsuya Ishikawa
        If W is generated from standard normal distribution, outputs z(x) where z(x)'z(y) approximates rbf_k(x, y)
        Returns n x D matrix
        :param X: a matrix of size n x d
        :param W: a matrix of size d x D//2
        """
        D = tf.dtypes.cast(W.shape[1], dtype=tf.float64)
        ts = tf.matmul(X, W)
        cs = tf.cos(ts)
        ss = tf.sin(ts)
        return tf.sqrt(1 / D) * tf.concat([cs, ss], axis=1)

    def batch_rff_z(X, W):
        """
        Returns tensor of shape (count, n, D)
        :param X: tensor of shape (count, n, d)
        :param W: tensor of shape (count, d, D//2)
        """
        D = tf.dtypes.cast(W.shape[-1], dtype=tf.float64)
        ts = tf.matmul(X, W)
        cs = tf.cos(ts)
        ss = tf.sin(ts)
        return tf.sqrt(1 / D) * tf.concat([cs, ss], axis=2)

    def W_theta(X, Y, D, count):
        """
        Returns W, theta where W are the weights for random Fourier features with shape (count, d, D//2) and theta
        are the associated posterior weights with shape (count, D, 1)
        :param X: input points of shape (n, d)
        :param Y: objective function evaluations of shape (n, 1)
        :param D: number of Fourier features to use
        :param count: number of Ws and thetas to generate
        """
        X = tf.tile(tf.expand_dims(X, axis=0), [count, 1, 1])  # (count, n, d)

        # Sample random features phi
        W = tf.random.normal(shape=(count, d, D // 2),
                             mean=0.0,
                             stddev=1.0,
                             dtype=tf.float64)
        phi = batch_rff_z(X, W)  # (count, n, D)

        # Sample posterior weights theta
        A = (tf.linalg.matrix_transpose(phi) @ phi) + tf.expand_dims(variance * tf.eye(D, dtype=tf.float64), axis=0)
        A_inv = tf.linalg.inv(A)  # (count, D, D)
        theta_mean = tf.squeeze(A_inv @ tf.linalg.matrix_transpose(phi) @ tf.expand_dims(y, axis=0))  # (count, n)
        theta_var = variance * A_inv
        theta_dist = tfp.distributions.MultivariateNormalFullCovariance(loc=theta_mean,
                                                                        covariance_matrix=theta_var)
        theta = tf.expand_dims(tf.dtypes.cast(theta_dist.sample(), dtype=tf.float64), axis=2)

        return (W, theta)

    def construct_maximizer_objective(x_star_latent, W, theta):
        # Construct g
        x_star = tf.sigmoid(x_star_latent)  # Constrains the value of x_star to between 0 and 1
        g = tf.reduce_sum(batch_rff_z(x_star, W) @ theta)
        return -g

    W, theta = W_theta(X, y, D, count)

    # Compute x_star using gradient based methods
    optimizer = tf.keras.optimizers.Adam()
    x_star_latent = tf.Variable(tf.random.normal(shape=(count, 1, d), dtype=tf.dtypes.float64))
    loss = lambda: construct_maximizer_objective(x_star_latent, W, theta)

    for i in range(num_steps):
        optimizer.minimize(loss, var_list=[x_star_latent])
        # if i % 200 == 0:
        #     print('Loss at step %s: %s' % (i, loss().numpy()))

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
def p_x_star_cond_D(x_star, model, num_samples=1000):
    """
    Among possible maximizers of the function, calculates the probability that each is the global maximizer.
    Samples from the model and calculates an expectation from the samples.
    Returns a tensor of shape (num_max)
    :param x_star: possible maximizers, tensor of shape (num_max, d)
    :param model: GPflow model
    """
    num_max = x_star.shape[0]

    samples = tf.squeeze(model.predict_f_samples(x_star, num_samples))  # (num_samples, num_max)
    samples_argmax = tf.argmax(samples, 1, output_type=tf.int32)  # (num_samples)
    count = tf.zeros(num_max, dtype=tf.float64) + tf.keras.backend.epsilon()  # Start all at one so we avoid dividing by zero later on
    count += tf.math.bincount(samples_argmax, minlength=num_max, dtype=tf.float64)
    return count / num_samples


@tf.function
def p_z_cond_D_x_star(z, x_star, p_x_star_cond, model, num_samples=1000):
    """
    Returns a tensor of shape (num_max)
    :param z: tuple of the form (x, chi) where chi is a tensor of shape (num_choices, d) and x is an index to
    the most preferred input in chi
    :param x_star: possible maximizers, tensor of shape (num_max, d)
    :param p_x_star_cond_D: tensor of shape (num_max)
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

    return (1. / p_x_star_cond) * (p_z_cond_f_z / num_samples)


def I(chi, x_star, model):
    """
    Predictive Entropy Search acquisition function.
    :param chi: input points in a query, tensor of shape (num_choices, d)
    :param x_star: possible maximizers, tensor of shape (num_max, d)
    :param model: GPflow model
    """

    num_choices = chi.shape[0]
    num_max = x_star.shape[0]

    p_x_star_cond = p_x_star_cond_D(x_star, model)  # (num_max)

    expected_log = tf.Variable(initial_value=tf.zeros((num_max, num_choices), dtype=tf.float64),
                               dtype=tf.float64,
                               shape=(num_max, num_choices))  # p(z|D,x_star)*log((p|D,x_star)/p(z|D))
    for i in range(num_choices):
        z = (i, chi)
        p_z_cond = p_z_cond_D_x_star(z, x_star, p_x_star_cond, model)
        norm_constant = tf.reduce_sum(p_x_star_cond * p_z_cond)
        expected_log[:, i].assign(p_z_cond * tf.math.log(p_z_cond / norm_constant))

    sum_over_preferred = tf.reduce_sum(expected_log, axis=1)

    return tf.reduce_sum(p_x_star_cond * sum_over_preferred)


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