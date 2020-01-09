"""
Given ordinal (preference) data consisting of sets of input points and a most preferred input point for every such set,
the train_model function learns variational parameters that approximate the distribution of a latent function f over
all input points present in the data, which can be used to construct GP models to approximate f over the entire input
space.
Formulation by Nguyen Quoc Phong.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow
from gpflow.utilities import set_trainable


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
    for the PBO formulation, in the case where inputs are independent.
    :param q_mu:
        Mean of variational distribution Q. Tensor with shape (num_data)
    :param q_var:
        Variance of variational distribution Q. Tensor with shape (num_data)
    :param D_idxs: tensor with shape (num_data, num_choices, 1)
        Input data points, that are indices into q_mu and q_var for tf.gather_nd
    :param max_idxs: tensor with shape (num_data, 1)
        Selection of most preferred input point for each collection of data points, that are indices into
        q_mu and q_var
    """
    return tf.reduce_sum(tf.gather_nd(q_mu, max_idxs)
                           - tf.math.log(tf.reduce_sum(tf.exp(
                                            tf.gather_nd(q_mu, D_idxs)
                                            + 0.5 * tf.gather_nd(q_var, D_idxs)
                                           ), axis=1) ) )


def elbo(p_mu, p_var, q_mu, q_var, D_idxs, max_idxs):
    """
    Calculates the ELBO for the PBO formulation.
    """
    return variational_expectations(q_mu, q_var, D_idxs, max_idxs) - kl_divergence(q_mu, q_var, p_mu, p_var)


def elbo_fullcov(q_mu, q_sqrt_latent, D_idxs, max_idxs, kernel, inputs):
    """
    Calculates the ELBO for the PBO formulation, using a full covariance matrix.
    :param q_mu: tensor with shape (num_data, 1)
    :param q_sqrt_latent: tensor with shape (1, num_data, num_data). Will be forced into lower triangular matrix such
    that q_sqrt @ q_sqrt^T represents the covariance matrix of X
    :param D_idxs: tensor with shape (num_data, num_choices, 1)
        Input data points, that are indices into q_mu and q_var for tf.gather_nd
    :param max_idxs: tensor with shape (num_data, 1)
        Selection of most preferred input point for each collection of data points, that are indices into
        q_mu and q_var
    :param kernel: gpflow kernel to calculate covariance matrix for KL divergence
    :param inputs: tensor of shape (num_data, input_dims) with indices corresponding to that of D_idxs and max_idxs
    """
    q_sqrt = tf.linalg.band_part(q_sqrt_latent, -1, 0) # Force into lower triangular
    q_sqrt_squeezed = tf.squeeze(q_sqrt, axis=0)
    q_full = q_sqrt_squeezed @ tf.transpose(q_sqrt_squeezed)
    q_diag = tf.linalg.diag_part(q_full)

    cov_mat = kernel.K(inputs)

    return variational_expectations(tf.squeeze(q_mu, axis=1), q_diag, D_idxs, max_idxs) \
           - gpflow.kullback_leiblers.gauss_kl(q_mu, q_sqrt, cov_mat)


def cholesky_matrix_inverse(A):
    """
    :param A: Symmetric positive-definite matrix, tensor of shape (n, n)
    :return: Inverse of A, tensor of shape (n, n)
    """
    L = tf.linalg.cholesky(A)
    L_inv = tf.linalg.triangular_solve(L, tf.eye(A.shape[0], dtype=tf.float64))
    return tf.linalg.matrix_transpose(L_inv) @ L_inv


def multivariate_normal_log_pdf(mean, covariance, x):
    """
    Calculates the log probability density of the point x in the multivariate normal distribution specified.
    :param mean: tensor of shape (n)
    :param covariance: tensor of shape (n, n)
    :param x: tensor of shape (n)
    :return: scalar
    """

    mahalanobis_squared = tf.squeeze(tf.expand_dims(x - mean, axis=0) @
                                     cholesky_matrix_inverse(covariance) @
                                     tf.expand_dims(x - mean, axis=1))
    return -0.5 * (mahalanobis_squared + tf.cast(x.shape[0] * tf.math.log(2 * np.pi), dtype=tf.float64) +
                   tf.linalg.logdet(covariance))


def q_joint_f_u(q_mu, q_sqrt_latent, inducing_variables, kernel, inputs):
    """
    Calculates the mean and covariance of the joint distribution q(f, u). Has form [f; u] where values corresponding
    to f values are before u values
    :param q_mu: tensor with shape (num_inducing, 1)
    :param q_sqrt_latent: tensor with shape (1, num_inducing, num_inducing). Will be forced into lower triangular matrix such
    that q_sqrt @ q_sqrt^T represents the covariance matrix of inducing variables
    :param inducing_variables: tensor with shape (num_inducing, input_dims)
    :param kernel: gpflow kernel to calculate covariance matrix for KL divergence
    :param inputs: tensor of shape (num_data, input_dims) with indices corresponding to that of D_idxs and max_idxs
    :return: (tensor of shape (num_data+num_inducing), tensor of shape (num_data+num_inducing, num_data+num_inducing))
    """
    m = inducing_variables.shape[0]  # num_inducing
    n = inputs.shape[0]  # num_data

    q_sqrt = tf.linalg.band_part(q_sqrt_latent, -1, 0)  # Force into lower triangular
    q_full = q_sqrt @ tf.linalg.matrix_transpose(q_sqrt)  # (1, num_data, num_data)

    Kmm = kernel.K(inducing_variables)  # (m, m)
    Kmm_inv = cholesky_matrix_inverse(Kmm)

    Knm = kernel.K(inputs, inducing_variables)  # (n, m)
    A = Knm @ Kmm_inv  # (n, m)

    # Build mean of joint distribution q(f, u) E[f; u] so f values stacked onto u values
    f_u_mean = tf.squeeze(tf.concat([A, tf.eye(m, dtype=tf.float64)], axis=0) @ q_mu, axis=-1)  # (n+m)

    # Build covariance of joint distribution q(f, u)
    Knn = kernel.K(inputs)
    S = tf.squeeze(q_full, axis=0)

    f_cov = Knn + (A @ (S - Kmm) @ tf.linalg.matrix_transpose(A))  # Marginal covariance of f
    f_u_cov = tf.concat([tf.concat([f_cov, S @ tf.linalg.matrix_transpose(A)], axis=0),
                         tf.concat([A @ S, S], axis=0)],
                        axis=1)

    return f_u_mean, f_u_cov


def elbo_inducing_variables(q_mu, q_sqrt_latent, inducing_variables, D_idxs, max_idxs, kernel, inputs):
    """
    Calculates the ELBO for the PBO formulation, using a full covariance matrix and inducing variables.
    :param q_mu: tensor with shape (num_inducing, 1)
    :param q_sqrt_latent: tensor with shape (1, num_inducing, num_inducing). Will be forced into lower triangular matrix such
    that q_sqrt @ q_sqrt^T represents the covariance matrix of inducing variables
    :param inducing_variables: tensor with shape (num_inducing, input_dims)
    :param D_idxs: tensor with shape (num_data, num_choices, 1)
        Input data points, that are indices into q_mu and q_var for tf.gather_nd
    :param max_idxs: tensor with shape (num_data, 1)
        Selection of most preferred input point for each collection of data points, that are indices into
        q_mu and q_var
    :param kernel: gpflow kernel to calculate covariance matrix for KL divergence
    :param inputs: tensor of shape (num_data, input_dims) with indices corresponding to that of D_idxs and max_idxs
    :return: tensor of shape ()
    """

    def log_likelihood_minus_KL(f_u):
        """
        Term that we calculate the expectation over the joint distribution q(f, u) for. Integrand for quadrature
        method
        """

        q_u_mean = tf.squeeze(q_mu, axis=-1)  # (num_inducing)
        q_sqrt = tf.linalg.band_part(q_sqrt_latent, -1, 0)  # Force into lower triangular
        q_u_cov = tf.squeeze(q_sqrt @ tf.linalg.matrix_transpose(q_sqrt), axis=0)  # (num_inducing, num_inducing)
        
        return


    f_u_mean, f_u_cov = q_joint_f_u(q_mu, q_sqrt_latent, inducing_variables, kernel, inputs)






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
    p_mu = tf.Variable(np.zeros(n), dtype=tf.float64)
    p_var = tf.Variable(np.ones(n), dtype=tf.float64)  # TODO: Change to gp kernel prior, change constant
    # Initialize variational parameters
    q_mu = tf.Variable(np.zeros(n), dtype=tf.float64)
    q_var = tf.Variable(np.ones(n), dtype=tf.float64)

    neg_elbo = lambda: -elbo(p_mu, p_var, q_mu, q_var, D_idxs, max_idxs)
    optimizer = tf.keras.optimizers.Adam()
    for i in range(num_steps):
        optimizer.minimize(neg_elbo, var_list=[q_mu, q_var])
        if i % 500 == 0:
            print('Negative ELBO at step %s: %s' % (i, neg_elbo().numpy()))

    inputs = np.array([idx_to_val_dict[i] for i in range(n)])

    return q_mu, q_var, inputs


def train_model_fullcov(X, y, num_steps=5000):
    idx_to_val_dict, val_to_idx_dict = populate_dicts(X)
    D_idxs, max_idxs = val_to_idx(X, y, val_to_idx_dict)

    n = len(val_to_idx_dict.keys())
    inputs = np.array([idx_to_val_dict[i] for i in range(n)])

    # Initialize variational parameters
    q_mu = tf.Variable(np.zeros([n, 1]), name="q_mu", dtype=tf.float64)
    q_sqrt_latent = tf.Variable(np.expand_dims(np.eye(n), axis=0), name="q_sqrt_latent", dtype=tf.float64)
    kernel = gpflow.kernels.RBF()
    kernel.lengthscale.assign(0.05)

    neg_elbo = lambda: -elbo_fullcov(q_mu, q_sqrt_latent, D_idxs, max_idxs,
                                     kernel=kernel,
                                     inputs=inputs)
    optimizer = tf.keras.optimizers.Adam()
    trainable_vars = [q_mu, q_sqrt_latent] + list(kernel.trainable_variables)
    for i in range(num_steps):
        optimizer.minimize(neg_elbo, var_list=trainable_vars)
        if i % 500 == 0:
            print('Negative ELBO at step %s: %s' % (i, neg_elbo().numpy()))

    return q_mu, tf.linalg.band_part(q_sqrt_latent, -1, 0), inputs, kernel  # q_mu and q_sqrt


def init_SVGP(q_mu, q_var, inputs, kernel, likelihood):
    """
    Returns a gpflow SVGP model using the values obtained from train_model.
    :param q_mu: np array or tensor of shape (num_inputs)
    :param q_var: np array or tensor of shape (num_inputs)
    :param inputs: np array or tensor of shape (num_inputs, input_dims)
    :param kernel: gpflow kernel
    :param likelihood: gpflow likelihood
    """

    model = gpflow.models.SVGP(kernel=kernel,
                               likelihood=likelihood,
                               inducing_variable=inputs)

    mu_vals = np.expand_dims(q_mu.numpy(), 1)
    model.q_mu.assign(mu_vals)

    # Transform q_var into diagonal matrix
    q_sqrt = np.sqrt(np.identity(len(q_var.numpy())) * np.outer(np.ones(len(q_var.numpy())), q_var.numpy()))
    model.q_sqrt.assign(np.expand_dims(q_sqrt, 0))

    # Set so that the parameters learned do not change if further optimization over
    # other parameters is performed
    set_trainable(model.q_mu, False)
    set_trainable(model.q_sqrt, False)
    set_trainable(model.inducing_variable.Z, False)

    return model


def init_SVGP_fullcov(q_mu, q_sqrt, inputs, kernel, likelihood):
    """
    Returns a gpflow SVGP model using the values obtained from train_model.
    :param q_mu: np array or tensor of shape (num_inputs, 1)
    :param q_sqrt: np array or tensor of shape (num_inputs, num_inputs). Lower triangular matrix
    :param inputs: np array or tensor of shape (num_inputs, input_dims)
    :param kernel: gpflow kernel
    :param likelihood: gpflow likelihood
    """

    model = gpflow.models.SVGP(kernel=kernel,
                               likelihood=likelihood,
                               inducing_variable=inputs)

    model.q_mu.assign(q_mu)
    model.q_sqrt.assign(q_sqrt)

    # Set so that the parameters learned do not change if further optimization over
    # other parameters is performed
    set_trainable(model.q_mu, False)
    set_trainable(model.q_sqrt, False)
    set_trainable(model.inducing_variable.Z, False)

    return model