import numpy as np
import tensorflow as tf

from typing import Tuple

from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import \
    Conv2D, Activation, MaxPooling2D, Flatten, Dense

from hw4_mnist import MNISTModel


class MNISTModelRegular(MNISTModel):
    """A version of an MNIST model to instrument the pre-activation output of the
       last intermediate layer (the one before the one that produces logits,
       the softmax layer has no trainable parameters) and adds L2
       regularization to all the parameters past this layer. We refer to the
       output of this layer as "features".
    """

    def __init__(self, lam=0.1, *argv, **kwargs):
        super().__init__(*argv, **kwargs)

        self.lam = lam

    def build(self):
        # Running this will reset the model's parameters
        layers = []

        # >>> Your code here <<<

        layers.append(Conv2D(
            filters=16,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="same",
            activation='relu'
        ))

        layers.append(MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2)
        ))

        layers.append(Conv2D(
            filters=32,
            kernel_size=(4, 4),
            padding='same',
            activation='relu'
        ))

        layers.append(MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2)
        ))

        layers.append(Flatten())

        # layers[-4]
        layers.append(Dense(64, activation='linear'))
        # Linear means no activation. Output are the features we are after.

        # layers[-3]
        layers.append(Activation("relu"))

        # layers[-2]
        # Last layer with trainable parameters. We add L2 regularization.
        layers.append(Dense(
            units=10,
            name="logits",
            kernel_regularizer=regularizers.l2(self.lam),
            bias_regularizer=regularizers.l2(self.lam)
        ))

        # layers[-1]
        layers.append(Activation('softmax'))

        # >>> End of your code <<<

        self.layers = layers

        self._define_ops()

    def forward(self, X, Ytrue=None):
        _features: tf.Tensor = None
        _logits: tf.Tensor = None
        _probits: tf.Tensor = None
        _preds: tf.Tensor = None
        _loss: tf.Tensor = None

        # >>> Your code here <<<

        c = X
        parts = []
        for l in self.layers:
            c = l(c)
            parts.append(c)

        _features = parts[-4]
        _logits = parts[-2]
        _probits = parts[-1]

        _preds = tf.argmax(_probits, axis=1)

        if Ytrue is not None:
            _loss = K.mean(K.sparse_categorical_crossentropy(
                self.Ytrue,
                _probits
            ))

        # >>> End of your code here <<<

        return {
            'features': _features,
            'logits': _logits,
            'probits': _probits,
            'preds': _preds,
            'loss': _loss,
        }

    def load(self, batch_size=16, filename=None):
        if filename is None:
            filename = f"model.regular{self.lam}.MNIST.h5"

        super().load(batch_size=batch_size, filename=filename)

    def save(self, filename=None):
        if filename is None:
            filename = f"model.regular{self.lam}.MNIST.h5"

        super().save(filename=filename)


class Representer(object):
    def __init__(
        self,
        model: MNISTModelRegular,
        X: np.ndarray,
        Ytrue: np.ndarray
    ) -> None:
        """
        X: np.ndarray [N, 28, 28, 1] training points
        Y: np.ndarray [N] ground truth labels
        """

        assert "features" in model.tensors, \
            "Model needs to provide features tensor."
        assert "loss" in model.tensors, \
            "Model needs to provide loss tensor."
        assert "logits" in model.tensors, \
            "Model needs to provide logits tensor."

        self.model = model
        self.lam = model.lam
        self.X = X
        self.Ytrue = Ytrue

        self._define_ops()

    def _define_ops(self):
        # >>> Your code here <<<

        self.f_feats = K.function(
            [self.model.X],
            self.model.tensors['features']
        )

        t_coeffs = 1 / (2 * self.model.lam * len(self.X)) \
            * K.gradients(
                    self.model.tensors['loss'],
                    self.model.tensors['logits']
              )[0]

        self.f_coeffs = K.function([self.model.X, self.model.Ytrue], t_coeffs)

        # >>> End of your code <<<

    def similarity(self, Xexplain: np.ndarray) -> np.ndarray:
        """For each input instance, compute the similarity between it and every one of
        the training instances. This is the f_i f_t^T term in the paper.

        inputs:
            Xexplain: np.ndarray [M, 28, 28, 1] -- images to explain

        return
            np.ndarray [M, N]

        """

        # >>> Your code here <<<

        feats_train = self.f_feats(self.X)
        feats_explain = self.f_feats(Xexplain)

        return np.moveaxis(np.matmul(feats_train, feats_explain.T), 0, 1)

        # >>> End of your code <<<

    def coeffs(self) -> np.ndarray:
        """For each training instance, compute its representer value coefficient. This
        is the alpha term in the paper.

        inputs:
            none

        return
            np.ndarray [N, 10]

        """

        # >>> Your code here <<<

        return self.f_coeffs([self.X, self.Ytrue])

        # >>> End of your code <<<

    def values(self, coeffs, sims) -> np.ndarray:
        """Given the training instance coefficients and train/test feature
        similarities, compute the representer point values. This is the k term
        from the paper.

        inputs:
            coeffs: np.ndarray [N, 10]
            sims: np.ndarray [M, N]

        return
            np.ndarray [M, N, 10]

        """

        # >>> Your code here <<<
        vals = np.expand_dims(coeffs, 2) \
            * np.expand_dims(np.moveaxis(sims, 0, 1), 1)
        # [N, 10, 1] * [N, 1, M]

        return np.moveaxis(vals, 2, 0)

        # >>> End of your code <<<

    def coeffs_and_values(
        self,
        Xexplain: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """For each input instance, compute the representer point coefficients of the
        training data(self.X, self.Y)

        inputs:
            Xexplain: np.ndarray [M, 28, 28, 1] -- images to explain
            target: target class being explained

        returns:
            coefficients: np.ndarray of size [N, 10]
            values: np.ndarray of size [M, N, 10]
              N is size of |self.X|

        """

        coeffs = self.coeffs()
        sims = self.similarity(Xexplain)

        return coeffs, self.values(coeffs, sims)
