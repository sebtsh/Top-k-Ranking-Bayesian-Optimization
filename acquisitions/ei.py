import numpy as np
from scipy.stats import norm


def EI(model, maximizer, input_points):
    """
    Expected improvement acquisition function by Mockus et al. (1978). Following Brochu (2010), this acquisition
    function over unary inputs is used in a pairwise query context by taking the incumbent maximizing point as the
    first input and the EI maximizer as the second input
    :param model: gpflow model
    :param maximizer: tensor of shape (1, input_dims). Incumbent maximizing point
    :param input_points: tensor of shape (num_discrete, input_dims). Calculates EI over these points
    :return: tensor of shape (num_discrete, 1)
    """
    num_discrete = input_points.shape[0]
    ei_vals = np.zeros((num_discrete, 1))

    f_max = np.squeeze(model.predict_f(maximizer)[0])
    f_mean, f_var = model.predict_f(input_points)
    f_mean = np.squeeze(f_mean, axis=1)
    f_var = np.squeeze(f_var, axis=1)
    f_stddev = np.sqrt(f_var)
    for i in range(num_discrete):
        if f_stddev[i] != 0:
            z = (f_mean[i] - f_max) / f_stddev[i]
            ei_vals[i] = (f_mean[i] - f_max) * norm.cdf(z) + f_stddev[i] * norm.pdf(z)

    return ei_vals
