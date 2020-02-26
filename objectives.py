import numpy as np


def forrester(x):
    """
    1-dimensional test function by Forrester et al. (2008)
    Defined as f(x) = (6x-2)^2 * sin(12x-4)
    :param x: tensor of shape (..., 1), x1 in [0, 1]
    :return: tensor of shape (..., )
    """
    x0 = x[..., 0]
    return (6 * x0 - 2) * (6 * x0 - 2) * np.sin(12 * x0 - 4)


def six_hump_camel(x):
    """
    2-D test function by Molga & Smutnicki (2005). Has 6 local minima, 2 of which are global.
     Has global minimum f(x) = -1.0316, at x = [0.0898, -0.7126] and x = [-0.0898, 0.7126]
    :param x: tensor of shape (..., 2), x1 in [-3, 3], x2 in [-2, 2]
    :return: tensor of shape (..., )
    """
    x1 = x[..., 0]
    x2 = x[..., 1]
    return (4. - 2.1 * (x1 ** 2) + (1. / 3.) * (x1 ** 4)) * (x1 ** 2) + (x1 * x2) + (-4. + 4 * (x2 ** 2)) * (x2 ** 2)


def hartmann3d(x):
    """
    3-D test function from Unconstrained Global Optimization Test Problems, at
    http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO.htm
    Has 4 local minima, one of which is global. Has global minimum f(x) = -3.86278, at x = [0.114614, 0.555649,
    0.852547]
    :param x: tensor of shape (..., 3), x in [0, 1] for all dimensions
    :return: tensor of shape (..., )
    """
    alpha = np.array([1., 1.2, 3.0, 3.2])
    A = np.array([[3.0, 10, 30],
                  [0.1, 10, 35],
                  [3.0, 10, 30],
                  [0.1, 10, 35]])
    P = 0.0001 * np.array([[3689, 1170, 2673],
                           [4699, 4387, 7470],
                           [1091, 8732, 5547],
                           [381, 5743, 8828]])

    x_repeat = np.repeat(np.expand_dims(x, axis=-2), 4, axis=-2)
    return -np.sum(alpha * np.exp(-np.sum(A * ((x_repeat - P) ** 2), axis=-1)), axis=-1)


def objective_get_y(x, objective):
    """
    Returns tensor with argmins of input array (input point with the lowest objective function value among choices)
    :param x: tensor of shape (..., num_choices, input_dims)
    :param objective: function that takes tensor of shape (..., input_dims) and outputs tensor of shape (..., ) with
    the objective function values
    :return: tensor of shape (..., input_dims)
    """
    evals = objective(x)  # (..., num_choices)
    indices = np.argmin(evals, axis=1)[:, np.newaxis, np.newaxis]  # (..., 1, 1)
    return np.squeeze(np.take_along_axis(x, indices, axis=-2), axis=-2)
