import numpy as np 
import scipy.stats as spst 
import scipy.special as spmc


NOISELESS = 0
GUMBLE = 1
GAUSS = 2
# LOGISTIC = 3

def gen_observation_from_f(x, fx, noise_model, noise_std = 0.1, indifference_threshold = 0.0):
    """
    param x: [num_data] list of 2d-array [:,d]
    param fx: [num_data] list of 1d-array [:,1]
    param noise_model: either NOISELESS or GUMBLE or GAUSS
    param noise_std: standard deviation of Gaussian noise (only for Gauss noise model)
    param indifference_threshold: threshold for indifference used for Gumble noise only
    """

    if noise_model == NOISELESS:
        selected_idxs = [preference_noiseless(fvals.squeeze()) for fvals in fx]

    elif noise_model == GUMBLE:
        selected_idxs = [preference_based_on_gumble(fvals.squeeze(), indifference_threshold) for fvals in fx]

    elif noise_model == GAUSS:
        selected_idxs = [preference_based_on_gauss(fvals.squeeze(), noise_std) for fvals in fx]
    
    # elif noise_model == LOGISTIC:
    #     # this is the same as GUMBLE noise model with num_choice = 2 and indifference_threshold = 0.0
    #     selected_idxs = [preference_based_on_logistic(fvals.squeeze()) for fvals in fx]
    
    else:
        raise Exception("Noise model should be either NOISELESS or GUMBLE or GAUSS")

    prefs = []
    for i, selected_idx in enumerate(selected_idxs):

        if selected_idx >= 0:
            prefs.append(x[i][selected_idx,:].reshape(1,-1))

        else: # indifferent
            prefs.append(None)

    return prefs 


def preference_noiseless(fvals):
    # :param fvals: 1d-array of size num_choice
    assert len(fvals.shape) == 1
    return np.argmax(fvals)



def preference_based_on_gumble(fvals, indifference_threshold=0.0):
    # :param fvals: 1d-array of size num_choice
    assert len(fvals.shape) == 1

    num_choice = fvals.shape[0]

    indifference_mat = (1.0 - np.eye(num_choice)) * indifference_threshold

    base_f = fvals.reshape(-1,1) + indifference_mat * indifference_threshold

    pref_prob = np.exp(fvals - spmc.logsumexp(base_f, axis=-2))
    indifference_prob = 1.0 - np.sum(pref_prob)
    
    prob = np.concatenate([pref_prob, [indifference_prob]], axis=0)
    cprob = np.cumsum(prob)

    t = np.random.rand()
    selected_idx = np.searchsorted(cprob, t)

    if selected_idx == num_choice:
        # indifferent
        selected_idx = -1

    return selected_idx


def preference_based_on_gauss(fvals, noise_std):
    # :param fvals: 1d-array of size 2
    assert len(fvals.shape) == 1 and fvals.shape[0] == 2

    z = (fvals[0] - fvals[1]) / np.sqrt(2) / noise_std
    prefer_zero_prob = spst.norm.cdf(z)
    
    t = np.random.rand()
    if t <= prefer_zero_prob:
        # prefer x[0]
        return 0
    else:
        # prefer x[1]
        return 1


# def preference_based_on_logistic(fvals):
#     # :param fvals: 1d-array of size 2
#     # the function implemented here satisfies p(x[0]) + p(x[1]) = 1
#     # where p(x[0]) is the probability that x[0] is preferred over x[1]
#     # NOTE: this is the same as GUMBLE noise model with num_choice = 2 and indifference_threshold = 0.0

#     assert len(fvals.shape) == 1 and fvals.shape[0] == 2

#     prefer_zero_prob = 1.0 / (1.0 + np.exp(fvals[1] - fvals[0]))

#     t = np.random.rand()
#     if t <= prefer_zero_prob:
#         # prefer x[0]
#         return 0
#     else:
#         # prefer x[1]
#         return 1



    