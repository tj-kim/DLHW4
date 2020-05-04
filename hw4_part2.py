# hw4_part2.py

from tqdm import tqdm

import numpy as np
import tensorflow as tf

import hw4_utils
from hw4_mnist import HW4Model, MNISTModel
from hw4_part1 import Attacker, PGDAttacker


def get_attacks(
    attacker,
    model,
    X, Y, n,
    target=None,
    batch_size=16
):
    """Generate attacks until I collect at least n."""

    ret_X = []
    ret_Y = []

    got = 0

    collecting = tqdm(
        unit="instances",
        total=n,
        desc="collecting attacks",
        leave=False,
    )

    while got < n:
        data = hw4_utils.Dataloader(
            X=X,
            y=Y,
            shuffle=True,
            batch_size=16
        )

        for X_batch, Y_batch in data:
            Xadv_batch = attacker.attack_batch(X_batch, Y_batch)
            predadv_batch = model.f_preds(Xadv_batch)

            if target is None:
                success = predadv_batch != Y_batch
            else:
                success = (Y_batch != target)*(predadv_batch == target)

            ret_X.append(Xadv_batch[success])
            ret_Y.append(Y_batch[success])

            got += success.sum()

            collecting.update(success.sum())

            if got >= n:
                break

    collecting.close()

    ret_X = np.vstack(ret_X)
    ret_Y = np.concatenate(ret_Y)

    return ret_X[0:n], ret_Y[0:n]


class FineTunable(object):
    def __init__(self, finetune: bool = False):
        self.finetune = finetune

    def defend(self) -> None:
        if not self.finetune:
            # If we are not finetuning, we are training from scratch.
            self.model.build() # resets the model


class Defender(object):
    def __init__(
        self,
        attacker: Attacker,
        model: HW4Model,
        batch_size: int = 16,
        epochs: int = 2,
    ) -> None:

        self.attacker = attacker
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs

    def defend(self) -> None:
        pass


class AugmentDefender(Defender, FineTunable):
    def __init__(
        self,
        finetune: bool = False,
        *argv, **kwargs
    ) -> None:
        """
            finetune: bool -- finetune the existing model instead of training
              from scratch
        """
        Defender.__init__(self, *argv, **kwargs)
        FineTunable.__init__(self, finetune)

    def defend(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        augment_ratio: float = 0.1,
    ):
        """Defend by augmenting the training data.

        inputs:
          X: np.ndarray - input images [N, 28, 28, 1]
          Y: np.ndarray - ground truth classes [N]

          augment_ratio: float -- how much adversarial data to use as a ratio
            of training data

        returns Xadv, Yadv (the adversarial instances generated in defense),
        self.model should be defended
          Xadv: np.ndarray [N*augment_ratio, 28, 28, 1]
          Yadv: np.ndarray [N*augment_ratio]

        """

        Xadv = None # the adversarial instances generated in the process of
                    # defense
        Yadv = None # and their (correct) classs

        # >>> Your code here <<<

        # Generate | X | * augment_ratio adversarial examples,

        Xadv, Yadv = get_attacks(
            self.attacker,
            self.model, X, Y,
            int(len(X)*augment_ratio),
            target=self.attacker.target
        )

        FineTunable.defend(self)

        # Resets model if not finetuning. If not finetuning, make sure you
        # generate the adversarial examples before you call this.

        new_X = Xadv
        new_Y = Yadv

        if not self.finetune:
            new_X = np.vstack([Xadv, X])
            new_Y = np.concatenate([Yadv, Y])

        self.model.train(new_X, new_Y, epochs=self.epochs)

        # >>> End of your code <<<

        return Xadv, Yadv


class PreMadryDefender(Defender, FineTunable):
    def __init__(self, finetune: bool = False, *argv, **kwargs) -> None:
        Defender.__init__(self, *argv, **kwargs)
        FineTunable.__init__(self, finetune)

    def defend(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ):
        """
        Defend by augmenting the training data.

        inputs:
          X: np.ndarray - input images [N, 28, 28, 1]
          Y: np.ndarray - ground truth classes [N]

        returns Xadv, Yadv (the adversarial instances generated in defense),
        self.model should be defended
          Xadv: np.ndarray [N*epochs, 28, 28, 1]
          Yadv: np.ndarray [N*epochs]
        """

        FineTunable.defend(self) # resets model if not finetuning

        Xadv = None # the adversarial instances generated in the process of
                    # defense
        Yadv = None # and their (correct) class

        # >>> Your code here <<<

        # For each input batch, generate adversarial examples and train on them
        # instead of original data.

        ret_X = []
        ret_Y = []

        for epoch in tqdm(
                range(self.epochs),
                unit="epochs",
                desc="defense training",
                leave=False
        ):
            data = hw4_utils.Dataloader(X, Y, batch_size=self.batch_size)

            for batch_X, batch_Y in tqdm(
                data, unit="batches",
                total=len(X)//self.batch_size,
                desc="defense training",
                leave=False
            ):
                batch_Xadv = self.attacker.attack_batch(batch_X, batch_Y)

                ret_X.append(batch_Xadv)
                ret_Y.append(batch_Y)

                self.model.model.train_on_batch(batch_Xadv, batch_Y)

        Xadv = np.vstack(ret_X)
        Yadv = np.concatenate(ret_Y)

        # >>> End of your code <<<

        return Xadv, Yadv


### The rest of this file is for a BONUS exercise.


class MNISTModelSymbolic(MNISTModel):
    def __init__(self, *argv, **kwargs) -> None:
        super().__init__(*argv, **kwargs)

    def build(self, input: tf.Tensor) -> None:
        # Now takes input as a tensor that might be the result of an attack.

        pass


class PGDAttackerSymbolic(PGDAttacker):
    def __init__(
        self,
        model: MNISTModelSymbolic,
        *argv, **kwargs
    ) -> None:
        super().__init__(model, *argv, **kwargs)

    def symbolic_attack(
        self,
        X: tf.Tensor,
        Y: tf.Tensor
    ) -> tf.Tensor:
        """ Symbolic attack. For BONUS exercise in Part II. """
        pass


class MadryDefender(Defender):
    def __init__(
            self,
            model: MNISTModelSymbolic,
            attacker: PGDAttackerSymbolic,
            *argv, **kwargs) -> None:
        super().__init__(model=model, attacker=attacker, *argv, **kwargs)

    def defend(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> None:
        """
        Defend by augmenting the training data.

        inputs:
          X: np.ndarray - input images [N, 28, 28, 1]
          Y: np.ndarray - ground truth classes [N]

        returns nothing; self.model should have been defended
        """

        super().defend() # resets model if not finetuning

        # >>> Your code here <<<

        # BONUS: Implement Madry's defense. You will need to extend/adjust the
        # model class and the attacker class to make the proper symbolic
        # connections.

        # >>> End of your code <<<
