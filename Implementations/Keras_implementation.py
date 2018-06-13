from keras.callbacks import Callback
from keras import backend as K
from pandas import DataFrame
import numpy as np


class RLNCallback(Callback):
    def __init__(self, layer, norm=1, avg_reg=-7.5, learning_rate=6e5):
        """
        An implementation of Regularization Learning, described in https://arxiv.org/abs/1805.06440, as a Keras
        callback.
        :param layer: The Keras layer to which we apply regularization learning.
        :param norm: Norm of the regularization. Currently supports only l1 and l2 norms. Best results were obtained
        with l1 norm so far.
        :param avg_reg: The average regularization coefficient, Theta in the paper.
        :param learning_rate: The learning rate of the regularization coefficients, nu in the paper. Note that since we
        typically have many weights in the network, and we optimize the coefficients in the log scale, optimal learning
        rates tend to be large, with best results between 10^4-10^6.
        """
        super(RLNCallback, self).__init__()
        self._kernel = layer.kernel
        self._prev_weights, self._weights, self._prev_regularization = [None] * 3
        self._avg_reg = avg_reg
        self._shape = K.transpose(self._kernel).get_shape().as_list()
        self._lambdas = DataFrame(np.ones(self._shape) * self._avg_reg)
        self._lr = learning_rate
        assert norm in [1, 2], "Only supporting l1 and l2 norms at the moment"
        self.norm = norm

    def on_train_begin(self, logs=None):
        self._update_values()

    def on_batch_end(self, batch, logs=None):
        self._prev_weights = self._weights
        self._update_values()
        gradients = self._weights - self._prev_weights

        # Calculate the derivatives of the norms of the weights
        if self.norm == 1:
            norms_derivative = np.sign(self._weights)
        else:
            norms_derivative = self._weights * 2

        if self._prev_regularization is not None:
            # This is not the first batch, and we need to update the lambdas
            lambda_gradients = gradients.multiply(self._prev_regularization)
            self._lambdas -= self._lr * lambda_gradients

            # Project the lambdas onto the simplex \sum(lambdas) = Theta
            translation = (self._avg_reg - self._lambdas.mean().mean())
            self._lambdas += translation

        # Clip extremely large lambda values to prevent overflow
        max_lambda_values = np.log(np.abs(self._weights / norms_derivative)).fillna(np.inf)
        self._lambdas = self._lambdas.clip_upper(max_lambda_values)

        # Update the weights
        regularization = norms_derivative.multiply(np.exp(self._lambdas))
        self._weights -= regularization
        K.set_value(self._kernel, self._weights.values.T)
        self._prev_regularization = regularization

    def _update_values(self):
        self._weights = DataFrame(K.eval(self._kernel).T)
