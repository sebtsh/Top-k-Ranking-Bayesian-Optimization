import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def fourier_features(X, W, b, kernel):
    """
    Given sampled tensors W and b, construct Fourier features of X
    :param X: tensor of shape (count, n, d)
    :param W: tensor of shape (count, D, d)
    :param b: tensor of shape (count, D, 1)
    :param kernel: gpflow kernel
    :return: tensor of shape (count, n, D)
    """
    D = W.shape[1]

    WX_b = W @ tf.linalg.matrix_transpose(X) + b  # (count, D, n)
    alpha = kernel.variance
    return tf.sqrt(2.0 * alpha / D) * tf.cos(tf.linalg.matrix_transpose(WX_b))  # (count, n, D)


def sample_fourier_features(X, kernel, D=100):
    """
    Fourier feature mapping of X for the RBF kernel
    :param X: tensor of shape (count, n, d)
    :return: tensor of shape (count, n, D)
    """
    count = X.shape[0]
    d = X.shape[2]

    W = tf.random.normal(shape=(count, D, d),
                         mean=0.0,
                         stddev=1.0 / kernel.lengthscale,
                         dtype=tf.float64)
    b = tf.random.uniform(shape=(count, D, 1),
                          minval=0,
                          maxval=2 * np.pi,
                          dtype=tf.float64)

    return fourier_features(X, W, b, kernel), W, b  # (count, n, D)


def sample_theta_variational(phi, q_mu, q_sqrt, likelihood_var):
    """
    Samples from distribution q(theta|D) = /int p(theta|y)p(y|f)q(f|D) df dy
    :param phi: Fourier features tensor with shape (count, n, D)
    :param q_mu: tensor of shape (n, 1)
    :param q_sqrt: tensor of shape (1, n, n). Lower triangular matrix
    :param likelihood_var: scalar. Variance of likelihood function p(y|f) from model
    :return: tensor with shape (count, D, 1)
    """
    n = phi.shape[1]
    D = phi.shape[2]

    q_var = q_sqrt @ tf.linalg.matrix_transpose(q_sqrt)  # (1, n, n)
    noise_I_D = tf.expand_dims(likelihood_var * tf.eye(D, dtype=tf.float64), axis=0)
    noise_I_n = tf.expand_dims(likelihood_var * tf.eye(n, dtype=tf.float64), axis=0)

    A = (tf.linalg.matrix_transpose(phi) @ phi) + noise_I_D
    A_inv = tf.linalg.inv(A)  # (count, D, D)
    M = A_inv @ tf.linalg.matrix_transpose(phi)  # (count, D, n)

    theta_mean = tf.squeeze(M @ tf.expand_dims(q_mu, axis=0), axis=-1)  # (count, D)
    theta_var = A_inv * likelihood_var + M @ (noise_I_n + q_var) @ tf.linalg.matrix_transpose(M)

    theta_dist = tfp.distributions.MultivariateNormalFullCovariance(loc=theta_mean,
                                                                    covariance_matrix=theta_var)
    theta = tf.expand_dims(tf.dtypes.cast(theta_dist.sample(), dtype=tf.float64), axis=-1)  # (count, D, 1)

    return theta


def sample_maximizers(X, count, n_init, D, model, min_val, max_val, num_steps=3000):
    """
    Samples from the posterior over the global maximizer using the method by Shah & Ghahramani (2015). Approximates
    the RBF kernel with its Fourier dual. Samples random Fourier features, constructs a linear model and computes
    the argmax using gradient-based optimization.

    :param X: input points, tensor with shape (n, d)
    :param count: number of maximizers to sample. Each will be taken from one separate function sample
    :param n_init: Number of initializing points for each function sample. This method will take the argmax of all
    initializing points after optimization, so that each function sample will have one maximizer returned
    :param D: number of Fourier features to use
    :param model: gpflow model that uses the RBF kernel and has been optimized
    :param min_val: float, min value that a maximizer can take
    :param max_val: float, max value that a maximizer can take
    :param num_steps: int that specifies how many optimization steps to take
    :return: tensor of shape (count, d)
    """
    d = X.shape[1]

    X = tf.tile(tf.expand_dims(X, axis=0), [count, 1, 1])  # (count, n, d)

    # Sample random features phi and get W and b
    phi, W, b = sample_fourier_features(X, model.kernel, D)  # phi has shape (count, n, D)

    # Sample posterior weights theta
    theta = sample_theta_variational(phi, model.q_mu, model.q_sqrt, model.likelihood.variance)

    def construct_maximizer_objective(x_star):
        g = tf.reduce_sum(fourier_features(x_star, W, b, model.kernel) @ theta)
        return -g

    # Compute x_star using gradient based methods
    optimizer = tf.keras.optimizers.Adam()
    x_star = tf.Variable(tf.random.uniform(shape=(count, n_init, d),
                                           minval=min_val,
                                           maxval=max_val,
                                           dtype=tf.dtypes.float64),
                         constraint=lambda x: tf.clip_by_value(x, min_val, max_val))
    loss = lambda: construct_maximizer_objective(x_star)

    prev_loss = loss().numpy()
    for i in range(num_steps):
        optimizer.minimize(loss, var_list=[x_star])
        current_loss = loss().numpy()
        if i % 500 == 0:
            print('Loss at step %s: %s' % (i, current_loss))
        if abs((current_loss-prev_loss) / prev_loss) < 1e-7:
            print('Loss at step %s: %s' % (i, current_loss))
            break
        prev_loss = current_loss

    return tf.reduce_max(x_star, axis=1)
