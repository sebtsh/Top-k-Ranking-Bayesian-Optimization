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


def sample_theta(phi, model, y):
    """

    :param phi: Fourier features tensor with shape (count, n, d)
    :param model: gpflow model
    :param y: objective function evaluations, tensor with shape (n, 1)
    :return: tensor with shape (count, D, 1)
    """
    D = phi.shape[-1]

    A = (tf.linalg.matrix_transpose(phi) @ phi) + \
        tf.expand_dims(model.likelihood.variance * tf.eye(D, dtype=tf.float64), axis=0)
    A_inv = tf.linalg.inv(A)  # (count, D, D)
    theta_mean = tf.squeeze(A_inv @ tf.linalg.matrix_transpose(phi) @ tf.expand_dims(y, axis=0))  # (count, D)

    theta_var = model.likelihood.variance * A_inv
    theta_dist = tfp.distributions.MultivariateNormalFullCovariance(loc=theta_mean,
                                                                    covariance_matrix=theta_var)
    theta = tf.expand_dims(tf.dtypes.cast(theta_dist.sample(), dtype=tf.float64), axis=2)  # (count, D, 1)

    return theta


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
    :return: tensor of shape (count, d)
    """
    d = X.shape[1]

    X = tf.tile(tf.expand_dims(X, axis=0), [count, 1, 1])  # (count, n, d)

    # Sample random features phi and get W and b
    phi, W, b = sample_fourier_features(X, model.kernel, D)  # phi has shape (count, n, D)

    # Sample posterior weights theta
    theta = sample_theta(phi, model, y)

    def construct_maximizer_objective(x_star_latent):
        # Construct g
        x_star = tf.sigmoid(x_star_latent)  # Constrains the value of x_star to between 0 and 1
        g = tf.reduce_sum(fourier_features(x_star, W, b, model.kernel) @ theta)
        return -g

    # Compute x_star using gradient based methods
    optimizer = tf.keras.optimizers.Adam()
    x_star_latent = tf.Variable(tf.random.normal(shape=(count, 1, d), dtype=tf.dtypes.float64))
    loss = lambda: construct_maximizer_objective(x_star_latent)

    for i in range(num_steps):
        optimizer.minimize(loss, var_list=[x_star_latent])
        if i % 5000 == 0:
            print('Loss at step %s: %s' % (i, loss().numpy()))

    return tf.squeeze(tf.sigmoid(x_star_latent), axis=1)