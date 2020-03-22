import numpy as np
import gpflow
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import os
import pickle

from gpflow.utilities import set_trainable, print_summary
gpflow.config.set_default_summary_fmt("notebook")

sys.path.append(os.path.split(os.path.split(os.getcwd())[0])[0]) # Move 2 levels up directory to import PBO
import PBO



objective = PBO.objectives.six_hump_camel
objective_low = -1.5
objective_high = 1.5
objective_dim = 2
objective_name = "SHC"
acquisition_name = "PES"
experiment_name = "PBO" + "_" + acquisition_name + "_" + objective_name


num_runs = 20
num_evals = 20
num_samples = 100
num_choices = 2
input_dims = 2
num_maximizers = 20
num_init_prefs = 3
num_inducing_init = 3
delta = 0.05 # Discretization of continuous input space
num_discrete_per_dim = int((objective_high - objective_low) / delta)



results_dir = os.getcwd() + '/tmp_results/' + experiment_name + '/'

try:
    # Create target Directory
    os.makedirs(results_dir)
    print("Directory " , results_dir ,  " created ") 
except FileExistsError:
    print("Directory " , results_dir ,  " already exists")


inputs = PBO.models.learning_fullgp.get_all_discrete_inputs(objective_low, objective_high, objective_dim, delta)
fvals = objective(inputs).reshape(num_discrete_per_dim, num_discrete_per_dim)


fig, ax = plt.subplots()
im = ax.imshow(fvals,
          interpolation="nearest",
         extent=(objective_low, objective_high, objective_low, objective_high),
         origin="lower",
         cmap="Spectral")
fig.colorbar(im, ax=ax)
plt.show()


def plot_gp(model, inducing_points, inputs, title, cmap="Spectral"):

    inputs = PBO.models.learning_fullgp.get_all_discrete_inputs(objective_low, objective_high, objective_dim, delta)
    predictions = model.predict_y(inputs)
    
    preds = tf.transpose(tf.reshape(predictions[0], [num_discrete_per_dim, num_discrete_per_dim]))
    variances = tf.transpose(tf.reshape(predictions[1], [num_discrete_per_dim, num_discrete_per_dim]))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(title)
    fig.set_size_inches(18.5, 6.88)
    fig.set_dpi((200))

    ax1.axis('equal')
    im1 = ax1.imshow(preds, 
                     interpolation='nearest', 
                     extent=(objective_low, objective_high, objective_low, objective_high), 
                     origin='lower', 
                     cmap=cmap)
    ax1.plot(inducing_points[:, 0], inducing_points[:, 1], 'kx', mew=2)
    ax1.plot(inputs[:, 0], inputs[:, 1], 'ko', mew=2, color='w')
    ax1.set_title("Mean")
    ax1.set_xlabel("x0")
    ax1.set_ylabel("x1")
    fig.colorbar(im1, ax=ax1)

    ax2.axis('equal')
    im2 = ax2.imshow(variances, 
                     interpolation='nearest', 
                     extent=(objective_low, objective_high, objective_low, objective_high), 
                     origin='lower', 
                     cmap=cmap)
    ax2.plot(inducing_points[:, 0], inducing_points[:, 1], 'kx', mew=2)
    ax2.plot(inputs[:, 0], inputs[:, 1], 'ko', mew=2, color='w')
    ax2.set_title("Variance")
    ax2.set_xlabel("x0")
    ax2.set_ylabel("x1")
    fig.colorbar(im2, ax=ax2)

    plt.savefig(fname=results_dir + title + ".png")
    plt.show()


def get_noisy_observation(X, objective):
    f = PBO.objectives.objective_get_f_neg(X, objective)
    return PBO.observation_model.gen_observation_from_f(X, f, 1)


def train_and_visualize(X, y, title, lengthscale_init=None, signal_variance_init=None):
    
    # Train model with data
    result = PBO.models.learning_fullgp.train_model_fullcov(
                        X, y, 
                        obj_low=objective_low,
                        obj_high=objective_high,
                        lengthscale_init=lengthscale_init,
                        signal_variance_init=signal_variance_init,
                        indifference_threshold=0.,
                        n_sample=1000,
                        deterministic=False,
                        num_steps=3000)
    
    q_mu = result['q_mu']
    q_sqrt = result['q_sqrt']
    u = result['u']
    inputs = result['inputs']
    k = result['kernel']
    
    likelihood = gpflow.likelihoods.Gaussian()
    model = PBO.models.learning.init_SVGP_fullcov(q_mu, q_sqrt, u, k, likelihood)
    u_mean = q_mu.numpy()
    inducing_vars = u.numpy()
    
    # Visualize model
    plot_gp(model, inducing_vars, inputs, title)
    
    return model, inputs, u_mean, inducing_vars



def best_guess(model):
    """
    Returns a GP model's best guess of the global maximum of f.
    """
    xx = PBO.models.learning_fullgp.get_all_discrete_inputs(objective_low, objective_high, objective_dim, delta)
    res = model.predict_f(xx)[0].numpy()
    return xx[np.argmax(res)]


num_data_at_end = int(num_init_prefs + num_evals)
X_results = np.zeros([num_runs, num_data_at_end, num_choices, input_dims])
y_results = np.zeros([num_runs, num_data_at_end, 1, input_dims])
best_guess_results = np.zeros([num_runs, num_evals, input_dims])

np.random.seed(0)

init_vals = np.zeros([num_runs, num_init_prefs, num_choices, input_dims])

for run in range(num_runs):
    for i in range(num_init_prefs):
        init_vals[run,i] = PBO.models.learning_fullgp.get_random_inputs(
                                objective_low, 
                                objective_high, 
                                objective_dim, 
                                delta,
                                size=num_choices,
                                with_replacement=False,
                                exclude_inputs=None)


lengthscale_init = None
signal_variance_init = None

for run in range(num_runs):
    print("Beginning run %s" % (run))
    
    X = init_vals[run]
    y = get_noisy_observation(X, objective)
    
    model, inputs, u_mean, inducing_vars = train_and_visualize(X, y, 
                                                        "Run_{}:_Initial_model".format(run),
                                                        lengthscale_init,
                                                        signal_variance_init)
    # save optimized lengthscale and signal variance for next iteration
    lengthscale_init = model.kernel.lengthscale.numpy()
    signal_variance_init = model.kernel.variance.numpy()
    
    for evaluation in range(num_evals):
        print("Beginning evaluation %s" % (evaluation)) 

        # Sample possible next queries
        samples = PBO.models.learning_fullgp.sample_inputs(inputs.numpy(), 
                                                           num_samples, 
                                                           num_choices, 
                                                           objective_low, 
                                                           objective_high,
                                                           delta)

        # Sample maximizers
        print("Evaluation %s: Sampling maximizers" % (evaluation))
        maximizers = PBO.fourier_features.sample_maximizers(X=inducing_vars,
                                                            count=num_maximizers,
                                                            n_init=10,
                                                            D=100,
                                                            model=model,
                                                            min_val=objective_low,
                                                            max_val=objective_high)
        print(maximizers)

        # Calculate PES value I for each possible next query
        print("Evaluation %s: Calculating I" % (evaluation))
        I_vals = PBO.acquisitions.pes.I_batch(samples, maximizers, model)

        # Select query that maximizes I
        next_idx = np.argmax(I_vals)
        next_query = samples[next_idx]
        print("Evaluation %s: Next query is %s with I value of %s" % (evaluation, next_query, I_vals[next_idx]))

        X = np.concatenate([X, [next_query]])
        # Evaluate objective function
        y = np.concatenate([y, get_noisy_observation(np.expand_dims(next_query, axis=0), objective)], axis=0)
        
        print("Evaluation %s: Training model" % (evaluation))
        model, inputs, u_mean, inducing_vars = train_and_visualize(X, y,  
                                                                   "Run_{}_Evaluation_{}".format(run, evaluation),
                                                                  lengthscale_init,
                                                                  signal_variance_init)
        # save optimized lengthscale and signal variance for next iteration
        lengthscale_init = model.kernel.lengthscale.numpy()
        signal_variance_init = model.kernel.variance.numpy()

        best_guess_results[run, evaluation, :] = best_guess(model)
        print("Best_guess f({}) = {}".format(
                best_guess_results[run, evaluation, :], 
                objective(best_guess_results[run, evaluation, :])))

    X_results[run] = X
    y_results[run] = y


pickle.dump((X_results, y_results, best_guess_results), open(results_dir + "Xybestguess.p", "wb"))

def dist(x, y):
    """
    x and y have shape (..., input_dims)
    """
    return np.sqrt(np.sum((x - y) * (x - y), axis=-1))


print("Minimum: f({}) = {}".format(global_min_x, global_min_f))

for i in range(best_guess_results.shape[0]):
    diff_from_min_x = dist(best_guess_results[i], global_min_x)
    diff_from_min_f = objective(best_guess_results[i]) - global_min_f
    
    x_axis = list(range(num_init_prefs+1, num_init_prefs+1+num_evals))
    
    fig, axs = plt.subplots(1,2, figsize=(12,6))

    axs[0].plot(x_axis, diff_from_min_x, 'kx', mew=2)
    axs[0].set_xticks(x_axis)
    axs[0].set_xlabel('Evaluations', fontsize=18)
    axs[0].set_ylabel('Best guess x-distance', fontsize=16)
    axs[0].set_title("Run %s" % i)
    
    axs[1].plot(x_axis, diff_from_min_f, 'kx', mew=2)
    axs[1].set_xticks(x_axis)
    axs[1].set_xlabel('Evaluations', fontsize=18)
    axs[1].set_ylabel('Best guess f-distance', fontsize=16)
    axs[1].set_title("Run %s" % i)
    
    plt.show()

