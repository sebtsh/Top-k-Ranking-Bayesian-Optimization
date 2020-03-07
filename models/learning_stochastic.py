# todo
#     1. use samples to compute the objective function 
#         elbo_fullcov(q_mu, q_sqrt_latent, inducing_variables, D_idxs, max_idxs, kernel, inputs)
#        then it can works for different number of choices
#     2. val_to_idxs, populate_dicts: need to adaptive to different number of choices
#         by converting some to list
#     2. check function q_f to see if the implementation is correct
#         checked: correct
"""
    forester_get_Y:
        as X is a list, change the function!
    sample maximizers:
        change observations from inducing input, inducing variables
        to distribution of inducing variables

Given ordinal (preference) data consisting of sets of input points and a most preferred input point for every such set,
the train_model function learns variational parameters that approximate the distribution of a latent function f over
all input points present in the data, which can be used to construct GP models to approximate f over the entire input
space.
Formulation by Nguyen Quoc Phong.
"""

import time 

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import gpflow
from gpflow.utilities import set_trainable


def elbo_fullcov(q_mu, 
                q_sqrt_latent, 
                inducing_inputs, 
                D_idxs, 
                max_idxs, 
                kernel, 
                inputs, 
                indifference_threshold, 
                n_inducing_sample=50, 
                n_f_given_inducing_sample=30):
    """
    Calculates the ELBO for the PBO formulation, using a full covariance matrix.
    :param q_mu: tensor with shape (num_inducing, 1)
    :param q_sqrt_latent: tensor with shape (1, num_inducing, num_inducing). Will be forced into lower triangular
        matrix such that q_sqrt @ q_sqrt^T represents the covariance matrix of inducing variables
    :param inducing_inputs: tensor with shape (num_inducing, input_dims)
    :param D_idxs: tensor with shape (num_data, num_choices, 1)
        Input data points, that are indices into q_mu and q_var for tf.gather_nd
    :param max_idxs: tensor with shape (num_data, 1)
        Selection of most preferred input point for each collection of data points, that are indices into
        q_mu and q_var
    :param kernel: gpflow kernel to calculate covariance matrix for KL divergence
    :param inputs: tensor of shape (num_inputs, input_dims) with indices corresponding to that of D_idxs and max_idxs
    :return: tensor of shape ()
    """
    Kmm = kernel.K(inducing_inputs)
    
    logdet_Kmm = tf.linalg.logdet(Kmm)
    invKmm = cholesky_matrix_inverse(Kmm)

    num_data = D_idxs.size()

    # 1. Sample from q(u)
    standard_mvn = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(tf.shape(q_mu)[0], dtype=tf.float64), 
            scale_diag=tf.ones(tf.shape(q_mu)[0], dtype=tf.float64))

    standard_mvn_samples = standard_mvn.sample(n_inducing_sample)
    # (n_inducing_sample, num_inducing)

    q_sqrt = tf.linalg.band_part(q_sqrt_latent, -1, 0)  # Force into lower triangular
    q_full = q_sqrt @ tf.linalg.matrix_transpose(q_sqrt)  # (1, num_data, num_data)
    inv_q_full = cholesky_matrix_inverse(tf.squeeze(q_full, axis=0))

    posterior_inducing_samples = q_sqrt @ tf.expand_dims(standard_mvn_samples, axis=-1) + q_mu
    # (n_inducing_sample, num_inducing, 1)
    
    # 2. p(f|u) where u are samples from q(u)
    f_mean_given_inducing_sample, f_cov_given_inducing_sample = p_f_given_u(
            posterior_inducing_samples, 
            inducing_inputs, kernel, inputs, invKmm)
    # f_mean: (n_inducing_sample, num_inputs)
    # f_cov: (num_inputs, num_inputs)

    # 3. KL[q(u) || p(u)] = E_{q(u)} [log q(u)] - [log p(u)]
    zero_mean_inducing_samples = posterior_inducing_samples - q_mu
    # (n_inducing_sample, num_inducing, 1)

    klterm = -0.5 * tf.reduce_mean(
        tf.linalg.logdet(q_full) 
        + tf.transpose(zero_mean_inducing_samples, perm=[0,2,1]) @ inv_q_full @ zero_mean_inducing_samples
        - logdet_Kmm
        - tf.transpose(posterior_inducing_samples, perm= [0,2,1]) @ invKmm @ posterior_inducing_samples
    )

    def body(i, likelihood): 
        idxs = tf.squeeze(D_idxs.read(i))
        max_idx = max_idxs[i]
        num_choice = tf.shape(idxs)[0]

        fi_cov = tf.gather(
            tf.gather(f_cov_given_inducing_sample, indices=idxs, axis=0),
            indices=idxs, axis=1)
        # (num_choice, num_choice)

        fi_mean = tf.gather(f_mean_given_inducing_sample, indices=idxs, axis=1)
        # (n_inducing_sample, num_choice)
        
        standard_mvn_i = tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros_like(idxs, dtype=tf.float64), 
                scale_diag=tf.ones_like(idxs, dtype=tf.float64))

        standard_mvn_i_samples = standard_mvn_i.sample(n_f_given_inducing_sample)
        # (n_f_given_inducing_sample, num_choice)

        eigvalues, eigvects = tf.linalg.eigh(fi_cov)
        eigvalues = tf.clip_by_value(eigvalues, clip_value_min=0., clip_value_max=np.infty)
        transform_mat = eigvects @ tf.linalg.diag(tf.sqrt(eigvalues))

        zero_mean_f_samples = tf.squeeze(transform_mat @ tf.expand_dims(standard_mvn_i_samples, axis=-1), axis=-1)
        # (n_f_given_inducing_sample, num_choice)
        
        f_samples = tf.expand_dims(zero_mean_f_samples, axis=1) + fi_mean
        # (n_f_given_inducing_sample, n_inducing_sample, num_choice)

        # implementing a threshold for the choice of indifference
        mask_mat = tf.eye(num_choice, dtype=tf.float64)
        diff_mat = (1.0 - mask_mat) * indifference_threshold

        def true_fn(max_idx, f_samples, diff_mat):
            f_samples = f_samples + tf.gather(diff_mat, indices=max_idx, axis=0)

            return tf.reduce_mean(
                    tf.gather(f_samples, indices=max_idx, axis=-1) 
                    - tf.reduce_logsumexp(f_samples, axis=-1))

        def false_fn(f_samples, mask_mat, diff_mat):
            max_idx_f_samples = tf.squeeze(mask_mat @ tf.expand_dims(f_samples, axis=-1), axis=-1)
            # (n_f_given_inducing_sample, n_inducing_sample, num_choice)

            all_f_samples = tf.expand_dims(f_samples, axis=-1) + tf.gather(diff_mat, indices=max_idx, axis=0)
            # (n_f_given_inducing_sample, n_inducing_sample, num_choice, num_choice)

            all_choice_logprob = max_idx_f_samples - tf.reduce_logsumexp(all_f_samples, axis=-2)
            # (n_f_given_inducing_sample, n_inducing_sample, num_choice)

            indifference_prob = 1.0 - tf.exp( tf.reduce_logsumexp(all_choice_logprob, axis=-1) )
            indifference_prob = tf.clip_by_value(indifference_prob, clip_value_min=1e-50, clip_value_max=1.0 - 1e-50)
            indifference_logprob = tf.math.log(indifference_prob)
            # (n_f_given_inducing_sample, n_inducing_sample)
            
            return tf.reduce_mean( indifference_logprob )

        likelihood_i = tf.cond(max_idx >= 0, 
            lambda: true_fn(max_idx, f_samples, diff_mat), 
            lambda: false_fn(f_samples, mask_mat, diff_mat))

        likelihood = likelihood + likelihood_i

        return i+1, likelihood

    cond = lambda i, _: i < num_data

    _, likelihood = tf.while_loop(
            cond, 
            body, 
            (0, tf.constant(0.0, dtype=tf.float64)),
            parallel_iterations=10)

    elbo = likelihood - klterm

    return elbo


def cholesky_matrix_inverse(A):
    """
    :param A: Symmetric positive-definite matrix, tensor of shape (n, n)
    :return: Inverse of A, tensor of shape (n, n)
    """
    L = tf.linalg.cholesky(A)
    L_inv = tf.linalg.triangular_solve(L, tf.eye(A.shape[0], dtype=tf.float64))
    return tf.linalg.matrix_transpose(L_inv) @ L_inv


def p_f_given_u(inducing_vars, inducing_inputs, kernel, inputs, invKmm_prior):
    """
    Calculates the mean and covariance of the joint distriubtion p(f|u)
    :param inducing_vars: tensor with shape (nsample,num_inducing,1)
    :param inducing_inputs: tensor with shape (num_inducing, input_dims)
    :param kernel: gpflow kernel to calculate covariance matrix for KL divergence
    :param inputs: tensor of shape (num_inputs, input_dims) with indices corresponding to that of D_idxs and max_idxs
    :return: (tensor of shape (nsample,num_inputs), tensor of shape (num_inputs, num_inputs))
    """
    Knm = kernel.K(inputs, inducing_inputs)  # (n, m)
    A = Knm @ invKmm_prior  # (n, m)

    f_mean = tf.squeeze(A @ inducing_vars, axis=-1)
    # (nsample, num_inducing)

    Knn = kernel.K(inputs)
    f_cov = Knn - A @ tf.transpose(Knm)
    # (num_inputs, num_inputs)

    return f_mean, f_cov


def q_f(q_mu, q_sqrt_latent, inducing_variables, kernel, inputs):
    """
    Calculates the mean and covariance of the joint distribution q(f)
    :param q_mu: tensor with shape (num_inducing, 1)
    :param q_sqrt_latent: tensor with shape (1, num_inducing, num_inducing). Will be forced into lower triangular
        matrix such that q_sqrt @ q_sqrt^T represents the covariance matrix of inducing variables
    :param inducing_variables: tensor with shape (num_inducing, input_dims)
    :param kernel: gpflow kernel to calculate covariance matrix for KL divergence
    :param inputs: tensor of shape (num_inputs, input_dims) with indices corresponding to that of D_idxs and max_idxs
    :return: (tensor of shape (num_inputs), tensor of shape (num_inputs, num_inputs))
    """

    q_sqrt = tf.linalg.band_part(q_sqrt_latent, -1, 0)  # Force into lower triangular
    q_full = q_sqrt @ tf.linalg.matrix_transpose(q_sqrt)  # (1, num_data, num_data)

    Kmm = kernel.K(inducing_variables)  # (m, m)
    Kmm_inv = cholesky_matrix_inverse(Kmm)

    Knm = kernel.K(inputs, inducing_variables)  # (n, m)
    A = Knm @ Kmm_inv  # (n, m)

    f_mean = tf.squeeze(A @ q_mu, axis=-1)

    Knn = kernel.K(inputs)
    S = tf.squeeze(q_full, axis=0)
    f_cov = Knn + (A @ (S - Kmm) @ tf.linalg.matrix_transpose(A))

    return f_mean, f_cov


def populate_dicts(D_vals):
    """
    Populates dictionaries to assign an index to each value seen in the training data.
    :param D_vals: [k] list of 2-d ndarray [:,d] (to allow different num_choice for different observations)
    """
    idx_to_val_dict = {}
    val_to_idx_dict = {}

    D_vals_list = np.concatenate(D_vals, axis=0)
    D_vals_list_tuples = [tuple(i) for i in D_vals_list]
    D_vals_set = set(D_vals_list_tuples)

    for val in D_vals_set:
        val_to_idx_dict[val] = len(val_to_idx_dict)
        idx_to_val_dict[len(val_to_idx_dict)-1] = val

    return idx_to_val_dict, val_to_idx_dict


def val_to_idx(D_vals, max_vals, val_to_idx_dict):
    """
    Converts training data from real values to index format using dictionaries.
    Returns D_idxs (tensor with shape (k, num_choices, 1)) 
        and max_idxs (tensor with shape (k, 1)):
            max_idxs[i,0] is argmax of D_idxs[i,:,0]
    :param D_vals: [k] list of ndarray [:,d]
    :param max_vals: [k] list of ndarray [1,d]
    """

    k = len(D_vals)

    max_idxs = np.zeros(k, dtype=np.int32)

    for i in range(k):
        if max_vals[i] is not None:
            diff = np.sum(np.square(D_vals[i] - max_vals[i]), axis=1)
            max_idxs[i] = np.where(diff < 1e-30)[0]
        else:
            max_idxs[i] = -1

    max_idxs = tf.constant(max_idxs)

    D_idxs = tf.TensorArray(dtype=tf.int32, size=k, name='D_idxs', infer_shape=False, clear_after_read=False)
    
    for i in range(k):
        np.stack([ [val_to_idx_dict[tuple(datum)]] for datum in D_vals[i] ])

    cond = lambda i, _: i < k
    body = lambda i, D_idxs: \
        (i+1, 
         D_idxs.write(
            i, 
            tf.constant([ val_to_idx_dict[tuple(datum)] for datum in D_vals[i] ], dtype=tf.int32)
            )
        )
    
    _, D_idxs = tf.while_loop(cond, body, (0, D_idxs))

    return D_idxs, max_idxs


def train_model_fullcov(X,
                        y,
                        num_inducing,
                        obj_low,
                        obj_high,
                        lengthscale=1.,
                        num_steps=5000,
                        indifference_threshold=None):
    """
    if indifference_threshold is None:
        indifference_threshold is trained with maximum likelihood estimation
    else:
        indifference_threshold is fixed
    :param X: np array with shape (num_data, num_choices, input_dims). Ordinal data
    :param y: np array with shape (num_data, input_dims). Most preferred input for each set of inputs. Each y value must
    match exactly to one of the choices in its corresponding X entry
    :param num_inducing: number of inducing variables to use
    :param obj_low: int. Floor of possible inducing point value in each dimension
    :param obj_high: int. Floor of possible inducing point value in each dimension
    :param lengthscale: float. Lengthscale to initialize RBF kernel with
    :param num_steps: int that specifies how many optimization steps to take when training model
    :param indifference_threshold:
    """
    input_dims = X.shape[2]
    idx_to_val_dict, val_to_idx_dict = populate_dicts(X)
    D_idxs, max_idxs = val_to_idx(X, y, val_to_idx_dict)

    n = len(val_to_idx_dict.keys())
    inputs = np.array([idx_to_val_dict[i] for i in range(n)])

    # Initialize variational parameters

    q_mu = tf.Variable(np.zeros([num_inducing, 1]), name="q_mu", dtype=tf.float64)
    q_sqrt_latent = tf.Variable(np.expand_dims(np.eye(num_inducing), axis=0), name="q_sqrt_latent", dtype=tf.float64)
    kernel = gpflow.kernels.RBF(lengthscale=[lengthscale for i in range(input_dims)])

    # Ensure Kmm is positive semidefinite
    psd = False
    while not psd:
        u = tf.Variable(np.random.uniform(low=obj_low, high=obj_high, size=(num_inducing, input_dims)),
                        name="u",
                        dtype=tf.float64,
                        constraint=lambda x: tf.clip_by_value(x, obj_low, obj_high))
        try:
            Kmm = kernel.K(u)
            L = tf.linalg.cholesky(Kmm)
            psd = True
        except tf.errors.InvalidArgumentError as err:
            print(err)
            print("Kmm:")
            print(Kmm)
            print("Resampling inducing variables u")

    is_threshold_trainable = (indifference_threshold is None)

    if is_threshold_trainable:
        indifference_threshold = tf.Variable(0.1, dtype=tf.float64, 
                        constraint=lambda x: tf.clip_by_value(x, 
                                                clip_value_min=0.0, 
                                                clip_value_max=np.infty))

    neg_elbo = lambda: -elbo_fullcov(q_mu=q_mu,
                                     q_sqrt_latent=q_sqrt_latent,
                                     inducing_inputs=u,
                                     D_idxs=D_idxs,
                                     max_idxs=max_idxs,
                                     kernel=kernel,
                                     inputs=inputs,
                                     indifference_threshold=indifference_threshold,
                                     n_inducing_sample=50,
                                     n_f_given_inducing_sample=50)

    optimizer = tf.keras.optimizers.Adam()

    if is_threshold_trainable:
        print("Indifference_threshold is trainable.")
        trainable_vars = [q_mu, q_sqrt_latent, u, indifference_threshold] + list(kernel.trainable_variables)
    else:
        print("Indifference_threshold is fixed at {}".format(indifference_threshold))
        trainable_vars = [q_mu, q_sqrt_latent, u] + list(kernel.trainable_variables)

    start_time = time.time()

    for i in range(num_steps):
        optimizer.minimize(neg_elbo, var_list=trainable_vars)
        
        if i % 500 == 0:
            print('Negative ELBO at step {}: {} in {:.4f}s'.format(i, 
                       neg_elbo().numpy(), 
                       time.time() - start_time))
            
            start_time = time.time()

    if is_threshold_trainable:
        return q_mu, tf.linalg.band_part(q_sqrt_latent, -1, 0), u, inputs, kernel, indifference_threshold  # q_mu and q_sqrt
    else:
        return q_mu, tf.linalg.band_part(q_sqrt_latent, -1, 0), u, inputs, kernel  # q_mu and q_sqrt


def init_SVGP_fullcov(q_mu, q_sqrt, inducing_variables, kernel, likelihood):
    """
    Returns a gpflow SVGP model using the values obtained from train_model.
    :param q_mu: np array or tensor of shape (num_inputs, 1)
    :param q_sqrt: np array or tensor of shape (num_inputs, num_inputs). Lower triangular matrix
    :param inducing_variables: tensor of shape (num_inducing, input_dims)
    :param inputs: np array or tensor of shape (num_inputs, input_dims)
    :param kernel: gpflow kernel
    :param likelihood: gpflow likelihood
    """

    model = gpflow.models.SVGP(kernel=kernel,
                               likelihood=likelihood,
                               inducing_variable=inducing_variables,
                               whiten=False)

    model.q_mu.assign(q_mu)
    model.q_sqrt.assign(q_sqrt)

    # Set so that the parameters learned do not change if further optimization over
    # other parameters is performed
    set_trainable(model.q_mu, False)
    set_trainable(model.q_sqrt, False)
    set_trainable(model.inducing_variable.Z, False)

    return model
