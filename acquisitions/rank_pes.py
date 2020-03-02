"""
Contains functions for the Predictive Entropy Search acquisition function.
Formulation by Nguyen Quoc Phong.
    for top-k ranking
"""

import numpy as np
import scipy.special as spmc
import itertools


def I_batch(chi, x_star, model, topk=None, num_samples=10000, indifference_threshold = 0.0):
    """
    Predictive Entropy Search acquisition function.
    :param chi: input points in a query, tensor of shape (num_data, num_choice, d)
    :param x_star: possible maximizers, tensor of shape (num_max, d)
    :param model: GPflow model
    """
    num_data = chi.shape[0]
    num_choice = chi.shape[1]
    num_max = x_star.shape[0]
    d = x_star.shape[1]

    if topk is None:
        topk = num_choice

    x_all = np.concatenate([chi.reshape(num_data * num_choice, d), x_star.reshape(num_max, d)], axis=0)
    # (num_data * num_choice + num_max, d)

    fsample_all = model.predict_f_samples(x_all, num_samples).numpy().reshape(num_samples, num_data * num_choice + num_max)
    # (num_samples, num_data * num_choice + num_max)

    fstar_samples = fsample_all[:,(num_data*num_choice):]
    # (num_samples, num_max)

    fchi_samples = fsample_all[:,:(num_data*num_choice)]
    fchi_samples = fchi_samples.reshape(num_samples, num_data, num_choice)
    # (num_samples, num_data, num_choice)

    # 2. p(x_star|D)
    fstar_sample_argmax = np.argmax(fstar_samples, axis=1)
    count = np.bincount(fstar_sample_argmax)
    p_xstar = count / np.sum(count) # note: some xstar has 0 count
    # (num_max,)

    n_permutations = precompute_n_permutations(num_choice, topk)

    # 3. log p(z|D)
    n_total_permutation = n_permutations[topk] 
    n_max_permutation = 1000 # cap the number of permutations for computational time
    is_observation_randomized = (n_total_permutation > n_max_permutation)

    if is_observation_randomized:
        permutations = get_rand_permutation_k_in_n(num_choice, topk, n_max_permutation, n_permutations)
        # (n_max_permutation, topk)
    else:
        permutations = get_all_permutation_k_in_n(num_choice, topk, n_permutations)
        # (n_total_permutation, topk)

    log_p_obs = get_log_likelihood(fchi_samples, 
                        permutations, 
                        normalizer=num_samples)
    # (n_observation_type, num_data)

    n_observation_type = log_p_obs.shape[0] # min(n_max_permutation, n_total_permutation)

    print("Number of {} observations for top-{} in {} choices: {}".format(
            "randomized" if is_observation_randomized else "all possible",
            topk,
            num_choice, 
            n_observation_type))

    # 4. log p(z,xstar| D)
    log_p_xstar_obs = np.zeros([num_max, n_observation_type, num_data])

    for max_idx in range(num_max):
        if p_xstar[max_idx] > 0:

            group_sample_by_max_idx = np.where(fstar_sample_argmax == max_idx)[0]
            fchi_samples_by_max_idx = fchi_samples[group_sample_by_max_idx,:,:]
            
            log_p_xstar_obs[max_idx,:,:] = get_log_likelihood(fchi_samples_by_max_idx, 
                                        permutations, 
                                        normalizer=num_samples)
    
    # 5. p(z|D, xstar)
    # remove those xstar with no sample
    xstar_with_no_sample_idx = np.where(p_xstar < 1e-30)[0]

    p_xstar = np.delete(p_xstar, xstar_with_no_sample_idx, axis=0)
    # num_max1 is num_max after remove xstar_with_no_sample
    log_p_xstar = np.log(p_xstar)
    # (num_max1,)

    log_p_xstar_obs = np.delete(log_p_xstar_obs, xstar_with_no_sample_idx, axis=0)
    # (num_max1, n_observation_type, num_data)


    log_p_z_given_xstar = log_p_xstar_obs - log_p_xstar.reshape(-1,1,1)
    # (num_max1, n_observation_type, num_data)

    # 6. compute mutual information
    mutual_information = np.sum(
            np.sum(
            np.exp(log_p_xstar_obs) * (log_p_z_given_xstar - np.expand_dims(log_p_obs, axis=0)),
            axis=1),
        axis=0)
    # (num_data,)

    return mutual_information, log_p_xstar, log_p_obs, log_p_xstar_obs


def get_log_likelihood(fx, permutations, normalizer=None):
    """
    fx: (num_sample,num_data,num_choice)
    """
    num_sample = fx.shape[0]
    num_choice = fx.shape[2]
    num_data = fx.shape[1]

    if normalizer is None:
        normalizer = num_sample

    n_permutation = len(permutations)
    all_log_likelihood = np.zeros([n_permutation, num_data])

    for i,order in enumerate(permutations):
        log_likelihood = get_log_likelihood_given_order(
                                fx, 
                                order, 
                                normalizer)
        # (num_data)

        all_log_likelihood[i,:] = log_likelihood

    return all_log_likelihood


def get_log_likelihood_given_order(fx, order, normalizer=None):
    """
    order: indices of sorted preference (ascending) for a subset of choices (k in top-k)
        numpy array of size k

    given an order of preference of fx, compute the likelihood
        if indifference_threshold > 0.0
            consider all possible indifference observations
        else:
            don't consider indifference

    fx: (num_sample,num_data, num_choice)
    """
    num_sample = fx.shape[0]
    num_data = fx.shape[1]
    num_choice = fx.shape[2]
    k = order.shape[0]

    if normalizer is None:
        normalizer = num_sample

    log_likelihood = np.zeros([num_sample, num_data])

    choice_idxs = np.array(list(set(range(num_choice)).difference(set(order))), dtype=int)

    for i in range(k):
        choice_idxs = np.concatenate([[order[i]], choice_idxs], axis=0)
        selected_idx = order[i]

        log_likelihood += fx[:,:,selected_idx] - spmc.logsumexp(fx[:,:,choice_idxs], axis=-1)
        # (num_sample, num_data)

    """
    p(x0 > x1 > x2|fx) = p(x0 > x1 and x0 > x2|fx) p(x1 > x2|fx)
    p(x0 > x1 > x2) = sum_fx p(x0 > x1 > x2|fx)
    """
    log_likelihood = spmc.logsumexp(log_likelihood - np.log(normalizer), axis=0)
    # (num_data)

    return log_likelihood
    # (num_data)


def get_ith_permutation_k_in_n(n, k, i, n_permutations):
    # return the i-th subset of size k in a set of size n
    # n! / (n-k)!
    # n_permutations[i]: choose i in n
    # n_permutations[k]: choose k in n 

    options = list(range(n))
    x = np.zeros(k, dtype=int)
    idx = i

    for j in reversed(range(0,k)):

        opt_i = int(idx / n_permutations[j])

        x[k-1-j] = options[opt_i]
        del options[opt_i]

        idx -= opt_i * n_permutations[j]

    return x


def precompute_n_permutations(n, k):
    assert k <= n

    n_permutations = np.ones(k+1, dtype=int)
    # n_permutations[i]: choose i in n-(k-i)
    #   = factorials[n-(k-i)] / factorials[n-(k-i)-i]
    #   = factorials[n-k+i] / factorials[n-k]
    #   = n_permutations[i-1] * (n-k+i) / (n-k)
    n_permutations[1] = n-(k-1)

    for i in range(2,k+1):
        n_permutations[i] = n_permutations[i-1] * (n-k+i)
    
    return n_permutations


def get_all_permutation_k_in_n(n, k, n_permutations = None):
    assert k <= n

    if n_permutations is None:
        n_permutations = precompute_n_permutations(n, k)
    permutations = np.zeros([n_permutations[k], k], dtype=int)

    for i in range(n_permutations[k]):
        permutations[i,:] = get_ith_permutation_k_in_n(n, k, i, n_permutations)

    return permutations
    
    
def get_rand_permutation_k_in_n(n, k, size, n_permutations = None):
    assert k <= n 

    if n_permutations is None:
        n_permutations = precompute_n_permutations(n, k)
    permutations = np.zeros([size, k], dtype=int)
    
    randi = np.random.randint(low=0, high=n_permutations[k], size=size)
    for i,j in enumerate(randi):
        permutations[i,:] = get_ith_permutation_k_in_n(n, k, j, n_permutations)

    return permutations
