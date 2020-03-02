"""
Contains functions for the Predictive Entropy Search acquisition function.
Formulation by Nguyen Quoc Phong.
    allow indifference only for top-1
    Example: (x0, x1, x2) there are 4 possible observations:
        1. x0 is the most preferred
        2. x1 is the most preferred
        3. x2 is the most preferred
        4. none of the above, i.e., cannot identify the most preferred
"""

import numpy as np
import scipy.special as spmc
import itertools


def I_batch(chi, x_star, model, num_samples=1000, indifference_threshold = 0.1):
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
    log_p_obs = get_log_likelihood(fchi_samples, 
                        normalizer=num_samples, 
                        indifference_threshold=indifference_threshold)
    # (num_choice + 1, num_data)
    
    n_observation_type = log_p_obs.shape[0] # num_choice + 1

    # 4. log p(z,xstar| D)
    log_p_xstar_obs = np.zeros([num_max, n_observation_type, num_data])

    for max_idx in range(num_max):
        if p_xstar[max_idx] > 0:

            group_sample_by_max_idx = np.where(fstar_sample_argmax == max_idx)[0]
            fchi_samples_by_max_idx = fchi_samples[group_sample_by_max_idx,:,:]
            
            log_p_xstar_obs[max_idx,:,:] = get_log_likelihood(fchi_samples_by_max_idx, 
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


def get_log_likelihood(fx, normalizer=None, indifference_threshold = 0.0):
    """
    fx: (num_sample,num_data,num_choice)
    """
    num_sample = fx.shape[0]
    num_choice = fx.shape[2]
    num_data = fx.shape[1]

    if normalizer is None:
        normalizer = num_sample

    mask = np.ones(num_choice)
    indiff_mat = (1.0 - mask) * indifference_threshold
    
    fx = np.expand_dims(fx, axis=-1)
    # (num_sample, num_data, num_choice, 1)

    selected_fx = np.sum(fx * mask, axis=-2)
    base_fx = fx + indiff_mat

    log_likelihood_preference = selected_fx - spmc.logsumexp(base_fx, axis=-2)
    # (num_sample, num_data, num_choice)

    prob_indifference = 1.0 - np.sum(np.exp(log_likelihood_preference), axis=-1, keepdims=True)
    prob_indifference = np.clip_by_value(prob_indifference, a_min=1e-100, a_max=1.0 - 1e-100)
    log_likelihood_indifference = np.log(prob_indifference)
    # (num_sample, num_data, 1)

    log_likelihood = np.concatenate([log_likelihood_preference, log_likelihood_indifference], axis=-1)
    # (num_sample, num_data, num_choice + 1)
    
    log_likelihood = spmc.logsumexp( log_likelihood - np.log(normalizer), axis=0 )
    # (num_data, num_choice + 1)

    return log_likelihood.T 
    # (num_choice + 1, num_data)

