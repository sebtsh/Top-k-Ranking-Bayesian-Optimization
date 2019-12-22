"""
Given ordinal (preference) data consisting of sets of input points and a most preferred input point for every such set,
the train_model function learns variational parameters that approximate the distribution of a latent function f over
all input points present in the data, which can be used to construct GP models to approximate f over the entire input
space.
Formulation by Nguyen Quoc Phong.
"""

import numpy as np
import tensorflow as tf

def kl_divergence(p_mu, p_var, q_mu, q_var):
    """
    Calculates the KL divergence of Q from P, where Q and P are univariate Gaussian distributions.
    :param p_mu:
        Mean of P.
    :param p_var:
        Variance of P.
    :param q_mu:
        Mean of Q.
    :param q_var:
        Variance of Q.
    """
    return tf.reduce_sum(0.5 * tf.math.log(q_var / p_var) + (p_var + tf.square(p_mu - q_mu)) / (2. * q_var) - 0.5)

def variational_expectations(q_mu, q_var, D_idxs, max_idxs):
    """
    Calculates
        \int q(f) (\log p(D|f)) df
    for the PBO formulation.
    :param q_mu:
        Mean of variational distribution Q.
    :param q_var:
        Variance of variational distribution Q.
    :param D_idxs: tensor with shape (num_data, num_choices, 1)
        Input data points, that are indices into q_mu and q_var for tf.gather_nd
    :param max_idxs: tensor with shape (num_data, 1)
        Selection of most preferred input point for each collection of data points, that are indices into
        q_mu and q_var
    """
    return tf.reduce_sum( tf.gather_nd(q_mu, max_idxs)
                           - tf.math.log( tf.reduce_sum( tf.exp(
                                            tf.gather_nd(q_mu, D_idxs)
                                            + 0.5 * tf.gather_nd(q_var, D_idxs)
                                           ), axis=1) ) )

def elbo(p_mu, p_var, q_mu, q_var, D_idxs, max_idxs):
    """
    Calculates the ELBO for the PBO formulation.
    """
    return variational_expectations(q_mu, q_var, D_idxs, max_idxs) - kl_divergence(q_mu, q_var, p_mu, p_var)


def populate_dicts(D_vals):
    """
    Populates dictionaries to assign an index to each value seen in the training data.
    :param D_vals: np.array with shape (k, num_choices, d)
    """
    idx_to_val_dict = {}
    val_to_idx_dict = {}

    k = D_vals.shape[0]
    num_choices = D_vals.shape[1]
    d = D_vals.shape[2]

    D_vals = np.reshape(D_vals, [k * num_choices, d])

    cur_idx = 0
    for val in D_vals:
        tup = tuple(val)  # Dictionaries cannot take np arrays as keys
        if tup not in val_to_idx_dict.keys():
            val_to_idx_dict[tup] = cur_idx
            idx_to_val_dict[cur_idx] = tup
            cur_idx += 1

    return idx_to_val_dict, val_to_idx_dict


def val_to_idx(D_vals, max_vals, val_to_idx_dict):
    """
    Converts training data from real values to index format using dictionaries.
    Returns D_idxs (tensor with shape (k, num_choices, 1)) and max_idxs (tensor with shape (k, 1))
    :param D_vals: np.array with shape (k, num_choices, d)
    :param max_vals: np.array with shape (k, d)
    """
    k = D_vals.shape[0]
    num_choices = D_vals.shape[1]

    D_idxs = np.zeros([k, num_choices, 1], dtype=np.int32)
    max_idxs = np.zeros([k, 1], dtype=np.int32)

    for i in range(k):
        for j in range(num_choices):
            D_idxs[i][j][0] = val_to_idx_dict[tuple(D_vals[i][j])]
        max_idxs[i][0] = val_to_idx_dict[tuple(max_vals[i])]

    return tf.constant(D_idxs), tf.constant(max_idxs)

def train_model(X, y, num_steps=5000):
    """
    Returns variational parameters q_mu and q_var (model's learned approximations of the distributions of
    f given the training data X and y), and the corresponding inputs
    :param X: np array with shape (num_data, num_choices, input_dims). Ordinal data
    :param y: np array with shape (num_data, input_dims). Most preferred input for each set of inputs. Each y value must
    match exactly to one of the choices in its corresponding X entry
    :param num_iterations: int that specifies how many optimization steps to take when training model
    """

    idx_to_val_dict, val_to_idx_dict = populate_dicts(X)
    D_idxs, max_idxs = val_to_idx(X, y, val_to_idx_dict)

    n = len(val_to_idx_dict.keys())
    # Assume prior of mean 0 and covariance 1 for each input point
    p_mu = tf.Variable(np.zeros(n), name="p_mu", dtype=tf.float64)
    p_var = tf.Variable(np.ones(n), name="p_var", dtype=tf.float64)
    # Initialize variational parameters
    q_mu = tf.Variable(np.zeros(n), name="q_mu", dtype=tf.float64)
    q_var = tf.Variable(np.ones(n), name="q_var", dtype=tf.float64)

    neg_elbo = lambda: -elbo(p_mu, p_var, q_mu, q_var, D_idxs, max_idxs)
    optimizer = tf.keras.optimizers.Adam()
    for i in range(num_steps):
        optimizer.minimize(neg_elbo, var_list=[q_mu, q_var])

    inputs = np.array([idx_to_val_dict[i] for i in range(q_mu.numpy().shape[0])])

    return q_mu, q_var, inputs