"""
Contains functions for the Predictive Entropy Search acquisition function.
Formulation by Nguyen Quoc Phong.
"""

import numpy as np
import tensorflow as tf


def sample_maximizers_simple(model, count, num_discrete_points):
    """
    Samples maximizers from the GP by sampling real values at each discrete point and finding the argmax
    :param model: gpflow model
    :param count: number of maximizers
    :param num_discrete_points: number of discrete points over input space to sample
    :return: tensor of shape (count, 1)
    """

    xx = np.linspace(0.0, 1.0, num_discrete_points).reshape(num_discrete_points, 1)
    samples = model.predict_f_samples(xx, count)  # (count, num_discrete_points)
    samples_argmax = np.squeeze(np.argmax(samples, axis=1), axis=1)
    return np.expand_dims(np.take(xx, samples_argmax), axis=1)


def sample_maximizers_discrete(model, count, data):
    """
    Samples maximizers from the GP by sampling values at each discrete point and finding the argmax
    :param model: gpflow model
    :param count: number of maximizers
    :param data: tensor of shape (num_data, input_dims). Discrete points to sample from
    :return: tensor of shape (count, input_dims)
    """

    samples = model.predict_f_samples(data, count)  # (count, num_data)
    samples_argmax = np.squeeze(np.argmax(samples, axis=1), axis=1)
    return np.take(data, samples_argmax, axis=0)


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
def log_p_x_star_cond_D(num_max, samples_argmax):
    """
    Among possible maximizers of the function, calculates the log probability that each is the global maximizer.
    Samples from the model and calculates an expectation from the samples.

    :param num_max: int, number of maximizers that samples_argmax indexes over
    :param samples_argmax: tensor of shape (num_samples)
    :return: Tensor of shape (num_max)
    """
    num_samples = samples_argmax.shape[0]

    count = tf.zeros(num_max, dtype=tf.float64) + tf.keras.backend.epsilon()  # Add epsilon to avoid dividing by 0
    count += tf.math.bincount(samples_argmax, minlength=num_max, dtype=tf.float64)
    return tf.math.log(count / (num_samples + num_max * tf.keras.backend.epsilon()))


@tf.function
def log_p_z_cond_D_x_star(num_max, samples, samples_x_star_argmax, z, log_p_x_star_cond):
    """
    Returns a tensor of shape (num_max)
    :param num_max: int, number of maximizers that samples_x_star_argmax indexes over
    :param samples: tensor of shape (num_samples, num_max + num_choices)
    :param z: tuple of the form (x, chi) where chi is a tensor of shape (num_choices, d) and x is an index to
    the most preferred input in chi
    :param log_p_x_star_cond_D: tensor of shape (num_max)
    """
    num_samples = samples.shape[0]

    p_z_cond_f_z = tf.scatter_nd(indices=tf.expand_dims(samples_x_star_argmax, axis=1),
                                 updates=samples_likelihood(z, samples[:, num_max:]),
                                 shape=tf.constant([num_max])) + tf.keras.backend.epsilon()

    return tf.math.log(p_z_cond_f_z / num_samples) - log_p_x_star_cond


def I(chi, x_star, model, num_samples=1000):
    """
    Predictive Entropy Search acquisition function.
    :param chi: input points in a query, tensor of shape (num_choices, d)
    :param x_star: possible maximizers, tensor of shape (num_max, d)
    :param model: GPflow model
    """

    num_choices = chi.shape[0]
    num_max = x_star.shape[0]

    # Sample from the model once, to use for all expectation approximations
    x_vals = tf.concat([x_star, chi], 0)  # f(x_star), f(z)
    samples = tf.squeeze(model.predict_f_samples(x_vals, num_samples))  # (num_samples, num_max + num_choices)
    samples_x_star = samples[:, :num_max]  # (num_samples, num_max)
    samples_x_star_argmax = tf.argmax(samples_x_star, 1, output_type=tf.int32)  # (num_samples)

    log_p_x_star_cond = log_p_x_star_cond_D(num_max, samples_x_star_argmax)  # (num_max)

    expected_log = tf.Variable(initial_value=tf.zeros((num_max, num_choices), dtype=tf.float64),
                               dtype=tf.float64,
                               shape=(num_max, num_choices))  # p(z|D,x_star)*log((p|D,x_star)/p(z|D))
    for i in range(num_choices):
        z = (i, chi)
        log_p_z_cond = log_p_z_cond_D_x_star(num_max, samples, samples_x_star_argmax, z, log_p_x_star_cond)
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


def sample_inputs_discrete(current_inputs, data, num_samples, num_choices):
    """
    Uniformly samples random inputs to query objective function. Sampled inputs must have
    existing data points among the choices, otherwise the learned function values for the
    input choices will be independent of the already learned function values for other data points.
    Returns np array of shape (num_samples*num_inputs, num_choices, input_dims)
    :param current_inputs: np array of shape (num_inputs, input_dims)
    :param data: np array of shape (num_data, input_dims). Combinations will be drawn from here
    :param num_samples: int, number of random values to permutate with existing inputs
    :param num_choices: int, number of choices in an input query
    """
    num_inputs = current_inputs.shape[0]
    input_dims = current_inputs.shape[1]
    num_data = data.shape[0]

    samples = np.zeros([num_samples * num_inputs, num_choices, input_dims])
    random_indices = np.random.choice(num_data, (num_samples, num_choices - 1))
    data_samples = np.take(data, random_indices, axis=0)  # (num_samples, num_choices - 1, input_dims)

    for i in range(num_inputs):
        for j in range(num_samples):
            cur_idx = i * num_samples + j
            samples[cur_idx, 0, :] = current_inputs[i]
            samples[cur_idx, 1:, :] = data_samples[j]

    return tf.constant(samples)
