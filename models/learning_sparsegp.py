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
                standard_mvn_samples=None,
                n_sample=1000):
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
    num_inducing = tf.shape(inducing_inputs)[0]
    num_input = tf.shape(inputs)[0]

    standard_mvn = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(num_input, dtype=tf.float64),
            scale_diag=tf.ones(num_input, dtype=tf.float64))

    standard_mvn_samples = standard_mvn.sample(n_sample)
    # (n_sample, num_inducing)

    q_sqrt = tf.linalg.band_part(q_sqrt_latent, -1, 0)  # Force into lower triangular
    q_full = q_sqrt @ tf.linalg.matrix_transpose(q_sqrt)  # (1, num_data, num_data)
    # inv_q_full = cholesky_matrix_inverse(tf.squeeze(q_full, axis=0))
    # logdet_q_full = tf.linalg.logdet(q_full)

    # q(f) = \int p(f|u)q(u)du
    f_mean, f_cov = q_f(q_mu, q_full, inducing_inputs, kernel, inputs, Kmm, invKmm)
    # f_mean: (num_input,)
    # f_cov: (num_input, num_input)

    if standard_mvn_samples is None:
        standard_mvn = tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros(num_input, dtype=tf.float64),
                scale_diag=tf.ones(num_input, dtype=tf.float64))
                
        standard_mvn_samples = standard_mvn.sample(n_sample)
        # (n_sample, num_input)

    # eigvalues, eigvects = tf.linalg.eigh(f_cov)
    # eigvalues = tf.clip_by_value(eigvalues, clip_value_min=0., clip_value_max=np.infty)
    # transform_mat = eigvects @ tf.linalg.diag(tf.sqrt(eigvalues))
    transform_mat = tf.linalg.cholesky(f_cov)
    # (num_input, num_input)

    zero_mean_f_samples = tf.squeeze(transform_mat @ tf.expand_dims(standard_mvn_samples, axis=-1), axis=-1)
    f_samples = zero_mean_f_samples + f_mean
    # (n_sample, num_input)
    
    # KL[q(u) || p(u)] = KL[q(f) || p(f)] = E_{q(f)} log [q(f) / p(f)]
    transform_mat_inv = tf.linalg.triangular_solve(transform_mat, tf.eye(num_input, dtype=tf.float64))
    f_cov_inv = tf.linalg.matrix_transpose(transform_mat_inv) @ transform_mat_inv
    # (num_input, num_input)

    logdet_f_cov = tf.linalg.logdet(f_cov)

    f_cov_prior = kernel.K(inputs)
    f_cov_prior_inv = cholesky_matrix_inverse(f_cov_prior)
    logdet_f_cov_prior = tf.linalg.logdet(f_cov_prior)

    klterm = tf.reduce_mean(
        -0.5 * (#num_input * tf.math.log(2.0 * np.pi) + 
                logdet_f_cov 
                + zero_mean_f_samples @ f_cov_inv @ tf.linalg.matrix_transpose(zero_mean_f_samples))
        +0.5 * (#num_input * tf.math.log(2.0 * np.pi) +
                logdet_f_cov_prior
                + f_samples @ f_cov_prior_inv @ tf.linalg.matrix_transpose(f_samples))
    )

    def body(i, likelihood):
        idxs = tf.squeeze(D_idxs.read(i))
        max_idx = max_idxs[i]
        num_choice = tf.shape(idxs)[0]

        fi_samples = tf.gather(f_samples, indices = idxs, axis=1)
        # (n_sample, num_choice)

        # implementing a threshold for the choice of indifference
        mask_mat = tf.eye(num_choice, dtype=tf.float64)
        diff_mat = (1.0 - mask_mat) * indifference_threshold

        def true_fn(max_idx, fi_samples, diff_mat):
            fi_samples = fi_samples + tf.gather(diff_mat, indices=max_idx, axis=0)

            return tf.reduce_mean(
                    tf.gather(fi_samples, indices=max_idx, axis=-1)
                    - tf.reduce_logsumexp(fi_samples, axis=-1))

        def false_fn(fi_samples, mask_mat, diff_mat):
            all_f_samples = tf.expand_dims(fi_samples, axis=-1) + tf.gather(diff_mat, indices=max_idx, axis=0)
            # (n_sample, num_choice, num_choice)

            all_choice_logprob = fi_samples - tf.reduce_logsumexp(all_f_samples, axis=-2)
            # (n_sample, num_choice)

            indifference_prob = 1.0 - tf.exp( tf.reduce_logsumexp(all_choice_logprob, axis=-1) )
            # (n_sample,)
            indifference_prob = tf.clip_by_value(indifference_prob, clip_value_min=1e-50, clip_value_max=1.0 - 1e-50)
            indifference_logprob = tf.math.log(indifference_prob)

            return tf.reduce_mean( indifference_logprob )

        likelihood_i = tf.cond(max_idx >= 0,
            lambda: true_fn(max_idx, fi_samples, diff_mat),
            lambda: false_fn(fi_samples, mask_mat, diff_mat))

        likelihood = likelihood + likelihood_i

        return i+1, likelihood

    cond = lambda i, _: i < num_data

    _, likelihood = tf.while_loop(
            cond,
            body,
            (0, tf.constant(0.0, dtype=tf.float64)),
            parallel_iterations=30)

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
    Calculates the mean and covariance of the joint distribution p(f|u)
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


def q_f(q_mu, q_full, inducing_variables, kernel, inputs, Kmm, Kmm_inv):
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


def init_inducing_vars(input_dims, num_inducing, obj_low, obj_high):
    """
    Initialize inducing variables. We create a uniform grid of points within the hypercube, then take num_inducing
    number of random points from that grid.
    :param input_dims: int
    :param num_inducing: int
    :param obj_low: float
    :param obj_high: float
    :return: tensor of shape (num_inducing, input_dims)
    """
    if input_dims == 1:
        return np.expand_dims(np.linspace(obj_low, obj_high, num_inducing + 2)[1:num_inducing + 2 - 1], axis=1)
    else:
        # Figure out minimum number of discrete per dim required
        num_discrete_per_dim = int(np.ceil(np.sqrt(num_inducing-1))) + 1
        num_points = num_discrete_per_dim ** input_dims
        grid = np.zeros([num_points, input_dims])
        discrete_points = np.linspace(obj_low, obj_high, num_discrete_per_dim + 2)[1: num_discrete_per_dim + 2 - 1]
        for i in range(num_points):
            for dim in range(input_dims):
                val = num_discrete_per_dim ** (dim)
                grid[i, dim] = discrete_points[int((i // val) % num_discrete_per_dim)]

        # Take num_inducing random points from grid
        indices = np.random.choice(num_points, num_inducing, replace=False)
        return np.take(grid, indices, axis=0)


def train_model_fullcov(X,
        y,
        num_inducing,
        obj_low,
        obj_high,
        deterministic=False,
        n_sample = 1000,
        lengthscale_lower_bound=gpflow.default_jitter(),
        num_steps=5000,
        indifference_threshold=0.0,
        inducing_vars=None,
        regularizer_lengthscale_mean_over_range=0.5,
        regularizer_lengthscale_std_over_range=0.35):
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
    :param lengthscale_prior: tensorflow_probability distribution
    :param num_steps: int that specifies how many optimization steps to take when training model
    :param indifference_threshold:
    """
    input_dims = X.shape[2]
    idx_to_val_dict, val_to_idx_dict = populate_dicts(X)
    D_idxs, max_idxs = val_to_idx(X, y, val_to_idx_dict)

    n = len(val_to_idx_dict.keys())
    inputs = np.array([idx_to_val_dict[i] for i in range(n)])
    num_input = inputs.shape[0]

    # Initialize variational parameters
    q_mu = tf.Variable(np.zeros([num_inducing, 1]), name="q_mu", dtype=tf.float64)
    q_sqrt_latent = tf.Variable(np.expand_dims(np.eye(num_inducing), axis=0), name="q_sqrt_latent", dtype=tf.float64)
    kernel = gpflow.kernels.RBF(lengthscale=[1.0 for i in range(input_dims)])

    if lengthscale_lower_bound is not None:
        kernel.lengthscale.transform = gpflow.utilities.bijectors.positive(lower=lengthscale_lower_bound)

    # if inducing_vars is None:
    #     inducing_vars = init_inducing_vars(input_dims, num_inducing, obj_low, obj_high)
    #
    # if isinstance(inducing_vars, np.ndarray):
    #     init_inducing_vars = inducing_vars
    # else:
    #     init_inducing_vars = inducing_vars.numpy()

    u = tf.Variable(inducing_vars,
                    name="u",
                    dtype=tf.float64,
                    constraint=lambda x: tf.clip_by_value(x, obj_low, obj_high))

    is_threshold_trainable = (indifference_threshold is None)

    if is_threshold_trainable:
        indifference_threshold = tf.Variable(0.1, dtype=tf.float64,
                        constraint=lambda x: tf.clip_by_value(x,
                                                clip_value_min=0.0,
                                                clip_value_max=np.infty))
    else:
        indifference_threshold = tf.constant(indifference_threshold, dtype=tf.float64)

    if deterministic:
        standard_mvn_samples = tf.constant(np.random.randn(n_sample, num_input), dtype=tf.float64)
    else:
        standard_mvn_samples = None

    input_range = obj_high - obj_low
    lengthscale_mean_regularizer = input_range * regularizer_lengthscale_mean_over_range
    lengthscale_std_regularizer = input_range * regularizer_lengthscale_std_over_range

    lengthscale_regularizer = 0.5 * tf.reduce_sum(tf.square((kernel.lengthscale.read_value() - lengthscale_mean_regularizer) / lengthscale_std_regularizer))

    neg_elbo = lambda: -elbo_fullcov(q_mu=q_mu,
                                     q_sqrt_latent=q_sqrt_latent,
                                     inducing_inputs=u,
                                     D_idxs=D_idxs,
                                     max_idxs=max_idxs,
                                     kernel=kernel,
                                     inputs=inputs,
                                     indifference_threshold=indifference_threshold,
                                     standard_mvn_samples=standard_mvn_samples,
                                     n_sample=n_sample) \
                        + lengthscale_regularizer

    # optimizer = tf.keras.optimizers.Adam()
    optimizer = tf.keras.optimizers.RMSprop(rho=0.0 if deterministic else 0.9)
    print("Optimizer config: ", optimizer.get_config())

    trainable_vars = [u, q_mu, q_sqrt_latent] + list(kernel.trainable_variables)
    
    if is_threshold_trainable:
        print("Indifference_threshold is trainable.")
        trainable_vars.append(indifference_threshold)


    start_time = time.time()
    lengthscale_init = np.array([lengthscale_mean_regularizer for i in range(input_dims)])
    kernel.lengthscale.assign(lengthscale_init)

    # reduce initial lengthscale if it is too big
    while True:

        is_lengthscale_too_big = False

        try:
            cur_neg_elbo = neg_elbo().numpy()
            is_lengthscale_too_big = (cur_neg_elbo > 1e10)
        except tf.errors.InvalidArgumentError as err:
            # lengthscale is too big that it causes numerical error
            is_lengthscale_too_big = True
        
        if not is_lengthscale_too_big:
            break

        lengthscale_init = np.array(lengthscale_init) * 0.8
        kernel.lengthscale.assign(lengthscale_init)

    print("Initialize lengthscale at {}".format(lengthscale_init))
    print("   Initial negative ELBO: {}".format(cur_neg_elbo))


    try:
        for i in range(num_steps):
            optimizer.minimize(neg_elbo, var_list=trainable_vars)

            if i % 100 == 0:
                print('Negative ELBO at step {}: {} in {:.4f}s'.format(i,
                           neg_elbo().numpy(),
                           time.time() - start_time))
                start_time = time.time()

    except tf.errors.InvalidArgumentError as err:
        print(err)
        print(q_mu)
        print(q_sqrt_latent)
        print(u)
        print(inputs)
        gpflow.utilities.print_summary(kernel)
        raise ValueError

    result = {"q_mu": q_mu,
              "q_sqrt": tf.linalg.band_part(q_sqrt_latent, -1, 0),
              "inputs": inputs,
              "u": u,
              "kernel": kernel,
              "indifference_threshold": indifference_threshold,
              "loss": neg_elbo().numpy()}

    return result


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
