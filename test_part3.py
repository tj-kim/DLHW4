# These are minimal tests just to make sure homework coding parts exist and
# little else.

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# from gradescope_utils.autograder_utils.decorators import weight
from hw4_utils import TestCase, run_tests
from hw4_mnist import MNISTModel, load_data


class Part3Tests(TestCase):
    def __init__(self, *args, **kwargs):
        self.data = load_data()

        self.module_name = "hw4_part3"
        TestCase.__init__(self, *args, **kwargs)

#    @weight(0)
    def test_mnistmodelregular_exists(self):
        """ MNISTModelRegular exists """

        self.assertHasAttribute(
            self.solution, "MNISTModelRegular"
        )
        model = self.solution.MNISTModelRegular(lam=0.1)

        self.assertHasAttribute(model, "build")
        model.build()

        self.assertHasAttribute(model, "tensors")

        self.assertIn("features", model.tensors)
        self.assertIn("logits", model.tensors)
        self.assertIn("probits", model.tensors)
        self.assertIn("loss", model.tensors)

#    @weight(0)
    def test_representer_exists(self):
        """ Representer exists """

        self.assertHasAttribute(
            self.solution, "Representer"
        )

        self.assertHasAttribute(
            self.solution, "MNISTModelRegular"
        )
        model = self.solution.MNISTModelRegular(lam=0.1)

        self.assertHasAttribute(model, "build")
        model.build()

        rep = self.solution.Representer(
            model=model,
            X=self.data.train.X,
            Ytrue=self.data.train.Y
        )

        self.assertHasAttribute(rep, "similarity")
        self.assertHasAttribute(rep, "coeffs")
        self.assertHasAttribute(rep, "values")
        self.assertHasAttribute(rep, "coeffs_and_values")


if __name__ == '__main__':
    run_tests(Part3Tests)
