"""
Contains functions for the Predictive Entropy Search acquisition function.
Formulation by Nguyen Quoc Phong.
"""

import numpy as np
import scipy.special as spmc
import itertools


def I_batch(chi, x_star, model, num_samples=1000, indifference_threshold = 0.0):
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

    # 3. log p(z|D)
    assert num_choice <= 6, "num_choice = 7 incurs 7!=5040 permutations, which is too large. This requires random sampling of permutations. For current implementation, we consider all possible permutations."
    permutations = list(itertools.permutations(range(num_choice)))  
    # n_observation_type = factorial(num_choice) # = len(permutations)

    if indifference_threshold > 1e-200: # i.e., not 0.0
        separators = gen_separator(num_choice)
        separator_weights = [get_generator_weight(s) for s in separators]
    else:
        separators = [ list(range(num_choice+1)) ] 
        separator_weights = [1.0]
        # only 1 separator for no indifference

    log_p_obs = get_log_likelihood(fchi_samples, 
                        permutations, 
                        separators, 
                        separator_weights,
                        normalizer=num_samples, 
                        indifference_threshold=indifference_threshold)
    # (n_observation_type, num_data)
    # where n_observation_type = factorial(num_choice) * num_separators
    #   = len(permutations) * num_separators
    n_observation_type = log_p_obs.shape[0]
    print("Number of possible observations for {} choices: {}".format(num_choice, n_observation_type))
    print("Number of permutations: {}, separators: {}".format(len(permutations), len(separators)))

    # 4. log p(z,xstar| D)
    log_p_xstar_obs = np.zeros([num_max, n_observation_type, num_data])

    for max_idx in range(num_max):
        if p_xstar[max_idx] > 0:

            group_sample_by_max_idx = np.where(fstar_sample_argmax == max_idx)[0]
            fchi_samples_by_max_idx = fchi_samples[group_sample_by_max_idx,:,:]
            
            log_p_xstar_obs[max_idx,:,:] = get_log_likelihood(fchi_samples_by_max_idx, 
                                        permutations, 
                                        separators, 
                                        separator_weights,
                                        normalizer=num_samples, 
                                        indifference_threshold=indifference_threshold)

    
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


def get_log_likelihood(fx, permutations, separators, separator_weights, normalizer=None, indifference_threshold = 0.0):
    """
    fx: (num_sample,num_data,num_choice)
    """
    num_sample = fx.shape[0]
    num_choice = fx.shape[2]
    num_data = fx.shape[1]

    if normalizer is None:
        normalizer = num_sample

    n_permutation = len(permutations)
    n_separator = len(separators)

    num_observation = n_separator * n_permutation
    all_log_likelihood = np.zeros([num_observation, num_data])

    for i,order in enumerate(permutations):
        log_likelihood = get_log_likelihood_given_order(
                                fx, 
                                order, 
                                separators, 
                                separator_weights, 
                                normalizer, 
                                indifference_threshold)
        # (num_separator, num_data)

        all_log_likelihood[(i*n_separator):((i+1)*n_separator),:] = log_likelihood

    return all_log_likelihood


def get_generator_weight(separator):
    """
    0 | 1 2 and 0 | 2 1 are treated the same
    as there is no preference (i.e., indifference) 
    between 1 and 2 in this case
    """
    return np.prod(spmc.factorial( np.diff(separator) ))


def gen_separator(num_choice):
    """
    gen_separator(3) returns [[0, 3], [0, 1, 3], [0, 1, 2, 3], [0, 2, 3]]
    [0,1,3] means index_0 < (less preferred to) (index_1, index_2) 
    where (index_1, index_2) means no preference (i.e., indifference) between index_1 and index_2
        denoted as: index_0 | index_1 index_2
    Hence the separators for num_choice = 3 are:
    [0, 3]:       |0 1 2|: no preference at all
    [0, 1, 3]:    |0|1 2|
    [0, 1, 2, 3]: |0|1|2|: fully-ordered
    [0, 2, 3]:    |0 1|2|
    """

    all_seps = []

    for sepint in range( 0, int(2**(num_choice-1)) ):
        j = 0
        last_digit = -1
        sep = []

        while sepint > 0:

            if sepint % 2 != last_digit:
                last_digit = sepint % 2
                sep.append(j)
            
            sepint = int(sepint / 2)
        
            j+= 1

        if j < num_choice:
            sep.append(j)
        
        sep.append(num_choice)

        all_seps.append(sep)

    return all_seps


def get_log_likelihood_given_order(fx, order, separators, separator_weights, normalizer=None, indifference_threshold = 0.0):
    """
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

    if normalizer is None:
        normalizer = num_sample


    all_log_likelihood = np.zeros([len(separators), num_data])

    for sep_idx, separator in enumerate(separators):
    
        log_likelihood = np.zeros(num_data)
        for i in range(1,len(separator)):

            previous, current = separator[i-1], separator[i]

            if current - previous > 1:
                # when no preference exists (i.e., indifferent)
                choice_idxs = order[previous:current]
                selected_idx = -1
            
                log_likelihood_i = get_log_likelihood_given_preference(
                                            fx, 
                                            choice_idxs, 
                                            selected_idx, 
                                            normalizer, 
                                            indifference_threshold)
                # (num_data,)

                log_likelihood += log_likelihood_i


            if current < num_choice:
                # when preference exists
                nxt = separator[i+1]

                log_likelihood_i = np.zeros([nxt - current, num_data])

                for idx,j in enumerate(order[current:nxt]):
                    choice_idxs = np.concatenate([ order[:current], [j] ], axis=0)
                    selected_idx = len(choice_idxs) - 1

                    log_likelihood_i[idx,:] = get_log_likelihood_given_preference(
                                            fx, 
                                            choice_idxs, 
                                            selected_idx, 
                                            normalizer, 
                                            indifference_threshold)
                    # (num_data,)

                log_likelihood += spmc.logsumexp(log_likelihood_i, axis=0)

        # due to duplicating counting of indifference case
        log_likelihood -= np.log(separator_weights[sep_idx])

        all_log_likelihood[sep_idx,:] = log_likelihood

    return all_log_likelihood
    # (num_separator, num_data)


def get_log_likelihood_given_preference(fx, choice_idxs, selected_idx, normalizer=None, indifference_threshold = 0.0):
    """
    fx: (num_sample,num_data, num_choice)
    choice_idxs: subset of range(num_choice)
    selected_idx: x[:,choice_idxs[ selected_idx ],:] is selected
    """
    num_choice = choice_idxs.shape[0]
    
    assert choice_idxs.shape[0] > 1, "need at least 2 choices for preference"

    num_sample = fx.shape[0]
    num_data = fx.shape[1]
    
    if normalizer is None:
        normalizer = num_sample

    fx = fx[:,:,choice_idxs]

    mask_mat = np.eye(num_choice)
    indifference_mat = (1.0 - mask_mat) * indifference_threshold

    if selected_idx >= 0:
        
        # add threshold for choices different from selected_idx
        indifference_vec = indifference_mat[selected_idx,:]
        fx = fx + indifference_threshold * indifference_vec

        max_f = fx[:,:,selected_idx] # (num_sample, num_data)

        log_likelihood = max_f - spmc.logsumexp(fx, axis=-1)
        # num_sample, num_data

    else:
        # indifference to all choices
        fx = np.extend_dims(fx, axis=-1)
        # (num_sample, num_data, num_choice,1)

        selected_f = fx * mask_mat
        base_f = fx + indifference_threshold * indifference_mat

        choice_log_likelihood = np.sum(selected_f, axis=-2) - spmc.logsumexp(base_f, axis=-2)
        # num_sample, num_data, num_choice
        all_choice_log_likelihood = spmc.logsumexp(choice_log_likelihood, axis=-1)
        # num_sample, num_data

        indifference_likelihood = np.clip(1.0 - np.exp(choice_log_likelihood), a_min=1e-30, a_max=1.0 - 1e-30)
        log_likelihood = np.log(indifference_likelihood)
        # num_sample, num_data

    # average over samples
    log_likelihood = spmc.logsumexp(log_likelihood - np.log(normalizer), axis=0)
    # num_data

    return log_likelihood
   



# TODO: test if gpflow can predict_f_samples of 2 x that are identical with identical samples?
# it doesn't return identical samples!!



# def get_log_likelihood_full(x, fx, sorted_idxs, normalizer=None):
#     """
#     x: (num_data, num_choice, xdim)
#     fx: (num_sample,num_data, num_choice)
#     sorted_idxs: (num_choice)
#     """
#     num_sample = fx.shape[0]
#     num_data = fx.shape[1]
#     num_choice = fx.shape[2]

#     if normalizer is None:
#         normalizer = num_sample

#     log_likelihood = np.zeros(num_data)

#     for max_idx in range(1, num_choice):
#         max_f = fx[:,:,sorted_idxs[max_idx]].reshape(num_samples, num_data, 1)
#         base_f = fx[:,:,sorted_idxs[:(max_idx+1)]].reshape(num_sample, num_data, max_idx+1)

#         log_likelihood_i = np.squeeze(max_f - spmc.logsumexp(base_f, axis=1, keepdims=True), axis=-1)
#         # num_sample, num_data

#         # average over samples
#         log_likelihood_i = spmc.logsumexp(log_likelihood_i - np.log(normalizer), axis=0)
#         # num_data

#         log_likelihood += log_likelihood_i
#         # (num_data)

#     return log_likelihood



# def get_log_likelihood(x, fx, sorted_idxs, normalizer=None):
#     """
#     x: (num_data, num_choice, xdim)
#     fx: (num_sample,num_data, num_choice)
#     sorted_idxs: (num_data, num_choice)
#     """
#     num_sample = fx.shape[0]
#     num_data = sorted_idxs.shape[0]
#     num_choice = sorted_idxs.shape[1]

#     if normalizer is None:
#         normalizer = num_sample

#     log_likelihood = np.zeros(num_data)

#     for max_idx in range(1, num_choice):
        
#         idx_i = np.tile(list(range(num_data)).reshape(-1,1), reps=(1,num_data)).flatten()
#         idx_j = sorted_idxs[:,:(max_idx+1)].flatten()

#         max_f = fx[:,list(range(num_data)), sorted_idxs[max_idx]].reshape(num_sample,num_data,1)
#         base_f = fx[:,idx_i, idx_j].reshape(num_sample,num_data,max_idx+1)

#         log_likelihood_i = np.squeeze(max_f - spmc.logsumexp(base_f, axis=1, keepdims=True), axis=-1)
#         # num_sample, num_data

#         # average over samples
#         log_likelihood_i = spmc.logsumexp(log_likelihood_i - np.log(normalizer), axis=0)
#         # num_data

#         log_likelihood += log_likelihood_i

#     return log_likelihood
   
"""
given a sorted idxs
the indifference can happen multiple times and between consecutive idxs
0 1 2 3 4 5
indifference: (0 1) (2 3 4) 5
or (0 1) 2 3 (4 5)
using list of lists:
    (0 1) (2) (3) (4 5)
how to generate sub lists:
    number of separators: 1,2,...,n-1 where n is number of choices
    choose n_separator in (n-1) space between choices
        + indifference all
"""

