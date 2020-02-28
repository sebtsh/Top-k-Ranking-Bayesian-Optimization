import numpy as np
from objectives import forrester, six_hump_camel, hartmann3d, objective_get_y


def test_forrester():
    assert np.isclose(np.squeeze(forrester(np.array([[1/3]]))), 0)
    assert np.isclose(np.squeeze(forrester(np.array([0.757]))), -6.0207070)
    test_outputs = forrester(np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]]))
    assert np.all(np.less(forrester(np.array([[0.757 for i in range(10)]])), test_outputs))


def test_six_hump_camel():
    assert np.isclose(np.squeeze(six_hump_camel(np.array([[0, 0]]))), 0)
    assert np.isclose(np.squeeze(six_hump_camel(np.array([[0, 1]]))), 0)
    assert np.isclose(np.squeeze(six_hump_camel(np.array([[1, 0]]))), 1.9 + 1/3)
    assert np.isclose(np.squeeze(six_hump_camel(np.array([[0.0898, -0.7126]]))), -1.0316, atol=1e-04)
    assert np.isclose(np.squeeze(six_hump_camel(np.array([[-0.0898, 0.7126]]))), -1.0316, atol=1e-04)
    test_outputs = six_hump_camel(np.array([[[0.3 * i, 0.2*j] for i in range(10)] for j in range(10)]))
    shc_min = six_hump_camel(np.array([[[0.0898, -0.7126] for i in range(10)] for j in range(10)]))
    assert np.all(np.less(shc_min, test_outputs))
    test_outputs = six_hump_camel(np.array([[[-0.3 * i, -0.2*j] for i in range(10)] for j in range(10)]))
    assert np.all(np.less(shc_min, test_outputs))


def test_hartmann3d():
    assert np.isclose(np.squeeze(hartmann3d(np.array([[0.114614, 0.555649, 0.852547]]))), -3.86278, atol=1e-06)
    test_outputs = hartmann3d(np.array([[[[0.1 * i, 0.1 * j, 0.1 * k]
                                          for i in range(10)]
                                         for j in range(10)]
                                        for k in range(10)]))
    hartmann_min = hartmann3d(np.array([[[[0.114614, 0.555649, 0.852547]
                                          for i in range(10)]
                                         for j in range(10)]
                                        for k in range(10)]))
    assert np.all(np.less(hartmann_min, test_outputs))


def test_objective_get_y():
    forr_arr = np.array([[[0.1],
                          [0.3],
                          [0.0897],
                          [0.7698]],
                         [[0.000020323],
                          [0.0230111],
                          [0.01232],
                          [0.8]]])
    forr_output = objective_get_y(forr_arr, forrester)
    assert forr_output.shape == (2, 1)
    assert np.equal(forr_output[0], 0.7698)
    assert np.equal(forr_output[1], 0.8)

    shc_arr = np.array([[[0.1, 0.2],
                     [0.3, 0.4],
                     [0.0897, -0.7126],
                     [-0.0898, 0.7126]],
                    [[-0.0898, 0.7126],
                     [3, 0.4],
                     [899, -0.7126],
                     [898, 0.7126]],
                    [[898, 231126],
                     [3, 0.4],
                     [899, -0.7126],
                     [898, 0.7126]]])  # (3, 4, 2)
    shc_output = objective_get_y(shc_arr, six_hump_camel)
    assert shc_output.shape == (3, 2)
    assert np.all(np.equal(shc_output[0], np.array([-0.0898, 0.7126])))
    assert np.all(np.equal(shc_output[1], np.array([-0.0898, 0.7126])))

    hartmann_arr = np.array([[[0.114614, 0.555649, 0.852547],
                              [0.1, 0.555649, 0.852547]],
                             [[0.2, 0.3, 0.4],
                              [0.9, 0.3, 0.1]],
                             [[0.114614, 0.555649, 0.852547],
                              [0.1, 555.649, 852.547]]])  # (3, 2, 3)
    hartmann_output = objective_get_y(hartmann_arr, hartmann3d)
    assert hartmann_output.shape == (3, 3)
    assert np.all(np.equal(hartmann_output[0], np.array([0.114614, 0.555649, 0.852547])))
    assert np.all(np.equal(hartmann_output[2], np.array([0.114614, 0.555649, 0.852547])))


