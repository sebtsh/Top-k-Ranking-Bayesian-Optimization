import numpy as np
import tensorflow as tf
import gpflow
import random
import pickle

from .MCCGP import MultiChannelConvGP


class DCGPLayer:
    """
    Assume no zero padding.
    """

    def __init__(self, input_shape, filter_side_length, stride, num_filters, inducing_vars, final=False):
        """
        :param input_shape: (H, W, C), for height, width, and channels.
        :param filter_side_length: int
        :param stride: int
        :param num_filters: int, number of output channels
        :param inducing_vars: tensor of shape (M, F * F)
        :param final: bool indicating if this is output layer or not
        """
        self.F = filter_side_length
        self.S = stride
        self.num_filters = num_filters
        (self.H, self.W, self.C) = input_shape
        self.final = final

        if (self.H - self.F) % self.S != 0:
            raise ValueError("Incompatible input height, filter side length and stride")
        elif (self.W - self.F) % self.S != 0:
            raise ValueError("Incompatible input width, filter side length and stride")

        self.H_out = (self.H - self.F) // self.S + 1
        self.W_out = (self.W - self.F) // self.S + 1

        if final:
            if not (self.H_out == 1 and self.W_out == 1):
                raise ValueError("For final layer, H_out and W_out should be 1. Check length parameters")

        # Initialize kernel list
        kernel_list = [gpflow.kernels.RBF(1 / self.C) for i in range(self.C)]

        # Initialize multi-channel convolutional GP
        self.gp = MultiChannelConvGP(kernel=gpflow.kernels.RBF(),
                                     kernel_list=kernel_list,
                                     likelihood=gpflow.likelihoods.Gaussian(),
                                     inducing_variable=inducing_vars,
                                     num_latent=self.num_filters)

    def reshape_input(self, inputs):
        """
        :param inputs: tensor with shape (N, H, W, C)
        :return: tensor with shape (N * H_out * W_out, C * F * F)
        """
        N = inputs.shape[0]

        patches = tf.image.extract_patches(images=inputs,
                                           sizes=[1, self.F, self.F, 1],
                                           strides=[1, 2, 2, 1],
                                           rates=[1, 1, 1, 1],
                                           padding='VALID')  # (N, H_out, W_out, F * F * C)

        patches = tf.reshape(patches,
                             [N * self.H_out * self.W_out, self.F, self.F, self.C])  # (N * H_out * W_out, F, F, C)

        patches = tf.transpose(patches, [0, 3, 1, 2])  # (N * H_out * W_out, C, F, F)

        return tf.reshape(patches, [N * self.H_out * self.W_out, self.C * self.F * self.F])

    def reshape_output(self, outputs):
        """
        :param outputs: tensor with shape (N * H_out * W_out, num_filters)
        :return: tensor with shape (N, H_out, W_out, num_filters)
        """
        return tf.reshape(outputs, [-1, self.H_out, self.W_out, self.num_filters])

    def sample(self, X):
        """
        Samples the output of this layer, using the re-parametrization trick (Kingma et al., 2015) to ensure
        that gradients of parameters can be computed as model parameters are now used in a deterministic manner
        :param X: tensor with shape (N * H_out * W_out, C * F * F)
        :return: tensor with shape (N * H_out * W_out, num_filters)
        """
        epsilon = np.random.randn(X.shape[0], self.num_filters)
        fmean, fvar = self.gp.predict_f(X)
        return fmean + epsilon * tf.sqrt(fvar)

    def forward(self, X):
        """
        Reshapes input, performs sampling with re-parametrization trick, and reshapes output. Used for both
        training and inference.
        :param X: tensor with shape (N, H, W, C)
        :param final: Set to true if output layer. Changes final shape to only rank 2
        :return: tensor with shape (N, H_out, W_out, num_filters) if final=False,
        else tensor with shape (N, num_filters)
        """
        outputs = self.reshape_output(self.sample(self.reshape_input(X)))
        if not self.final:
            return outputs
        else:
            return tf.squeeze(outputs, [1, 2])


class DCGP:
    """
    Model with more than 1 convolutional GP layer.
    """

    def __init__(self, num_layers, input_shape, filter_side_lengths, strides, nums_filters, nums_inducing, dataset):
        """
        :param input_shape: input_shape: (H, W, C), for height, width, and channels.
        :param filter_side_lengths: list of ints
        :param strides: list of ints
        :param nums_filters: list of ints. Last number must be intended output dims of entire model (e.g. 10 if doing
        classification with 10 classes)
        :param nums_inducing: number of inducing patches per layer
        :param dataset: tensor of shape (N, H, W, C). Intended dataset, used to initialize inducing variables
        """
        if len(filter_side_lengths) != num_layers:
            raise ValueError("Length of filter length list should be equal to number of layers")
        elif len(strides) != num_layers:
            raise ValueError("Length of strides list should be equal to number of layers")
        elif len(nums_filters) != num_layers:
            raise ValueError("Length of num_filter list should be equal to number of layers")

        # Initialize layers
        self.layers = []
        self.trainable_variables = []
        current_input_shape = input_shape
        for i in range(num_layers):
            is_final = False
            if i == num_layers-1:
                is_final = True

            F = filter_side_lengths[i]
            inducing_vars = sample_inducing_patches(dataset, nums_inducing[i], F)
            layer = DCGPLayer(input_shape=current_input_shape,
                              filter_side_length=F,
                              stride=strides[i],
                              num_filters=nums_filters[i],
                              inducing_vars=inducing_vars,
                              final=is_final)
            self.layers.append(layer)
            self.trainable_variables += layer.gp.trainable_variables

    def forward(self, X):
        """
        Passes inputs through all layers in the model.
        :param X: tensor with shape (N, H, W, C)
        :return: tensor with shape (N, num_classes)
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def elbo(self, X, y):
        """
        Doubly stochastic variational inference ELBO as presented by Salimbeni & Deisenroth (2017).
        :param X: tensor of shape (N, H, W, C). Input images
        :param y: tensor of shape (N, 1). Labels
        :return: lower bound of log likelihood
        """
        # Likelihood term
        likelihood = multiclass_likelihood(self.forward(X), y)

        # KL divergence term
        kl_divergence = 0
        for layer in self.layers:
            kl_divergence += layer.gp.prior_kl()

        return likelihood + kl_divergence

    def predict(self, X):
        """
        Prediction. Uses Gaussian mixture drawing S samples of final f values and taking the average.
        :param X: tensor of shape (N, H, W, C). Input images
        :return: tensor of shape (N), indicating class labels
        """
        S = 100

        outputs = self.forward(X)  # (N, num_classes)
        for i in range(S - 1):
            outputs += self.forward(X)
        outputs = outputs / S

        return tf.argmax(outputs, axis=1)

    def evaluate(self, X, y):
        """
        Evaluates model on test dataset and labels.
        :param X: tensor of shape (N, H, W, C). Test images
        :param y: tensor of shape (N, 1). Test labels
        :return: float, accuracy of model on test dataset
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("Inputs and labels numbers not equal")

        N = X.shape[0]
        results = tf.equal(self.predict(X), tf.squeeze(y, axis=[1]))  # (N) of bools
        return tf.math.count_nonzero(results).numpy() / N

    def train(self, num_epochs, batch_size, X, y, Xtest, ytest):
        """
        Training.
        :param num_epochs: int
        :param batch_size: int
        :param X: tensor of shape (N, H, W, C). Input images
        :param y: tensor of shape (N, 1). Labels
        :param Xtest: tensor of shape (Ntest, H, W, C). Test images
        :param ytest: tensor of shape (Ntest, 1). Test labels
        :return:
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("Inputs and labels numbers not equal")

        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset_batched = dataset.batch(batch_size)
        optimizer = tf.keras.optimizers.Adam()
        losses = []
        accuracies = []
        for epoch in range(num_epochs):
            print("Starting epoch {}".format(epoch))
            counter = 0
            for batch in dataset_batched:
                counter += 1
                with tf.GradientTape() as tape:
                    loss = -self.elbo(batch[0], batch[1])
                    variables = self.trainable_variables
                    gradients = tape.gradient(loss, variables)
                    optimizer.apply_gradients(zip(gradients, variables))
                    if counter % 100 == 0:
                        print("Loss at batch {}: {}".format(counter, loss))
                        losses.append(loss)
            accuracy = self.evaluate(Xtest, ytest)
            print("Accuracy after epoch {}: {}".format(epoch, accuracy))
            accuracies.append(accuracy)

        pickle.dump((losses, accuracies, self.trainable_variables), open("DCGPlossaccuracymodel.p", "wb"))
        return losses, accuracies


def sample_inducing_patches(dataset, num_samples, patch_side_length):
    """
    Initialize inducing patches from dataset images
    :param dataset: tensor of shape (N, H, W, C). Intended dataset, used to initialize inducing variables
    :param num_samples: int
    :param patch_side_length: int
    :return: tensor of shape (num_samples, patch_side_length ** 2)
    """
    H = dataset.shape[1]
    W = dataset.shape[2]
    C = dataset.shape[3]
    num_source_images = 10

    rand_images_list = random.choices(dataset, k=num_source_images)
    rand_images = np.zeros([num_source_images, H, W, C])
    for i in range(num_source_images):
        rand_images[i] = rand_images_list[i].numpy()

    patches = tf.image.extract_patches(rand_images,
                                       [1, 5, 5, 1],
                                       [1, 1, 1, 1],
                                       [1, 1, 1, 1],
                                       'VALID')
    patches = tf.transpose(tf.reshape(patches, [-1, patch_side_length, patch_side_length, C]), [0, 3, 1, 2])
    patches = tf.reshape(patches, [-1, patch_side_length ** 2])
    rand_patches_list = random.choices(patches, k=num_samples)
    rand_patches = np.zeros([num_samples, patch_side_length ** 2])
    for i in range(num_samples):
        rand_patches[i] = rand_patches_list[i].numpy()

    return rand_patches


def multiclass_likelihood(logits, labels):
    """
    Calculates sum log p(y|f) for the multiclass case
    :param logits: (N, num_classes)
    :param labels: (N, 1)
    :return: scalar
    """
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("Logits and labels dimensions not equal")

    N = logits.shape[0]
    log_softmax = tf.nn.log_softmax(logits, axis=1)  # (N, num_classes)

    enums = np.reshape(list(range(N)), [N, 1])
    enums_labels = np.concatenate([enums, labels], axis=1)

    return tf.reduce_sum(tf.gather_nd(log_softmax, enums_labels))
