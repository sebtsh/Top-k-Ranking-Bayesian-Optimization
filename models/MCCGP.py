import numpy as np
import tensorflow as tf

from gpflow import kullback_leiblers
from gpflow.base import Parameter
from gpflow.config import default_float
from gpflow.models.model import GPModel
from gpflow.utilities import positive, triangular
from gpflow.models.util import inducingpoint_wrapper
from gpflow.conditionals.util import base_conditional


class MultiChannelConvGP(GPModel):
    """
    Implementation of multi-channel convolutional GPs with separate inducing outputs for each channel.
    """
    def __init__(self,
                 kernel,
                 kernel_list,
                 likelihood,
                 inducing_variable,
                 *,
                 mean_function=None,
                 num_latent: int = 1,
                 q_diag: bool = False,
                 q_mu=None,
                 q_sqrt=None,
                 whiten: bool = False,
                 num_data=None):
        """
        - kernel, likelihood, inducing_variables, mean_function are appropriate
          GPflow objects
        - kernel_list is a list of kernels, one for each channel
        - num_latent is the number of latent processes to use, defaults to 1
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        - num_data is the total number of observations, defaults to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        # init the super class, accept args
        super().__init__(kernel, likelihood, mean_function, num_latent)
        self.kernel = None  # this should not be used
        self.kernel_list = kernel_list  # One kernel for each channel
        self.C = len(kernel_list)
        self.num_data = num_data
        self.q_diag = q_diag
        self.whiten = whiten
        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

        # init variational parameters
        num_inducing = len(self.inducing_variable) * self.C  # every channel has separate inducing outputs
        self._init_variational_parameters(num_inducing, q_mu, q_sqrt, q_diag)

    def _init_variational_parameters(self, num_inducing, q_mu, q_sqrt, q_diag):
        """
        Constructs the mean and cholesky of the covariance of the variational Gaussian posterior.
        If a user passes values for `q_mu` and `q_sqrt` the routine checks if they have consistent
        and correct shapes. If a user does not specify any values for `q_mu` and `q_sqrt`, the routine
        initializes them, their shape depends on `num_inducing` and `q_diag`.

        Note: most often the comments refer to the number of observations (=output dimensions) with P,
        number of latent GPs with L, and number of inducing points M. Typically P equals L,
        but when certain multioutput kernels are used, this can change.

        Parameters
        ----------
        :param num_inducing: int
            Number of inducing variables, typically refered to as M.
        :param q_mu: np.array or None
            Mean of the variational Gaussian posterior. If None the function will initialise
            the mean with zeros. If not None, the shape of `q_mu` is checked.
        :param q_sqrt: np.array or None
            Cholesky of the covariance of the variational Gaussian posterior.
            If None the function will initialise `q_sqrt` with identity matrix.
            If not None, the shape of `q_sqrt` is checked, depending on `q_diag`.
        :param q_diag: bool
            Used to check if `q_mu` and `q_sqrt` have the correct shape or to
            construct them with the correct shape. If `q_diag` is true,
            `q_sqrt` is two dimensional and only holds the square root of the
            covariance diagonal elements. If False, `q_sqrt` is three dimensional.
        """
        q_mu = np.zeros((num_inducing, self.num_latent)) if q_mu is None else q_mu
        self.q_mu = Parameter(q_mu, dtype=default_float())  # [M, P]

        if q_sqrt is None:
            if self.q_diag:
                ones = np.ones((num_inducing, self.num_latent), dtype=default_float())
                self.q_sqrt = Parameter(ones, transform=positive())  # [M, P]
            else:
                q_sqrt = [np.eye(num_inducing, dtype=default_float()) for _ in range(self.num_latent)]
                q_sqrt = np.array(q_sqrt)
                self.q_sqrt = Parameter(q_sqrt, transform=triangular())  # [P, M, M]
        else:
            if q_diag:
                assert q_sqrt.ndim == 2
                self.num_latent = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=positive())  # [M, L|P]
            else:
                assert q_sqrt.ndim == 3
                self.num_latent = q_sqrt.shape[0]
                num_inducing = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=triangular())  # [L|P, M, M]

    def Kuu(self):
        """
        Computes prior covariance between inducing patches.
        :param U: tensor with shape (M, D)
        :return: tensor of shape (C * M, C * M)
        """
        U = self.inducing_variable.Z
        return tf.linalg.LinearOperatorBlockDiag(
            [tf.linalg.LinearOperatorFullMatrix(kernel.K(U)) for kernel in self.kernel_list]).to_dense()

    def Kuf(self, X):
        """
        Computes prior covariance between inducing patches and data patches.
        :param U: tensor with shape (M, D)
        :param X: tensor with shape (N, C * D), where C is number of channels and D is number of dimensions in a patch
        :return: tensor with shape (C * M, N)
        """
        U = self.inducing_variable.Z
        CD = X.shape[1]
        D = CD // self.C
        return tf.concat([self.kernel_list[i].K(U, X[:, i*D:(i+1)*D]) for i in range(self.C)],
                         axis=0)

    def Kff(self, X, full_cov=False):
        """
        Computes prior covariance between patches.
        :param X: tensor with shape (N, C * D), where C is number of channels and D is number of dimensions in a patch
        :return: tensor with shape (N, N)
        """
        N = X.shape[0]
        CD = X.shape[1]
        D = CD // self.C

        if full_cov:
            cov = tf.zeros([N, N], dtype=tf.float64)
            for i in range(self.C):
                cov += self.kernel_list[i].K(X[:, i*D:(i+1)*D])
        else:
            cov = tf.zeros([N], dtype=tf.float64)
            for i in range(self.C):
                cov += self.kernel_list[i].K_diag(X[:, i*D:(i+1)*D])
        return cov

    def prior_kl(self):
        return kullback_leiblers.gauss_kl(q_mu=self.q_mu,
                                          q_sqrt=self.q_sqrt,
                                          K=self.Kuu())

    def predict_f(self, Xnew: tf.Tensor, full_cov=False, full_output_cov=False) -> tf.Tensor:
        q_mu = self.q_mu
        q_sqrt = self.q_sqrt
        mu, var = base_conditional(Kmn=self.Kuf(Xnew),
                                   Kmm=self.Kuu(),
                                   Knn=self.Kff(Xnew),
                                   function=q_mu,
                                   q_sqrt=q_sqrt)
        return mu + self.mean_function(Xnew), var
