# These are minimal tests just to make sure homework coding parts exist and
# little else.

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# from gradescope_utils.autograder_utils.decorators import weight
import tensorflow as tf
from hw4_utils import TestCase, run_tests
from hw4_mnist import MNISTModel, load_data


class Part1Tests(TestCase):
    def __init__(self, *args, **kwargs):
        self.model = MNISTModel()
        self.model.load()
        self.data = load_data()

        self.module_name = "hw4_part1"
        TestCase.__init__(self, *args, **kwargs)

#    @weight(0)
    def test_CWL2_exists(self):
        """ CWL2 exists. """

        self.assertHasAttribute(
            self.solution, "CWL2Attacker"
        )

        att = self.solution.CWL2Attacker(
            model=self.model,
            num_steps=10,
            target=None,
            learning_rate=8.0,
            learning_rate_decay=0.9,
            k=0.0,
            c=4.0
        )

        self.assertHasAttribute(att, "loss")
        self.assertIsInstance(att.loss, tf.Tensor)

        self.assertHasAttribute(att, "step")
        self.assertIsInstance(att.step, list)

        self.assertHasAttribute(att, "output")
        self.assertIsInstance(att.output, tf.Tensor)


if __name__ == '__main__':
    run_tests(Part1Tests)
