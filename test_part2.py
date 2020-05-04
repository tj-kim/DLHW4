# These are minimal tests just to make sure homework coding parts exist and
# little else.

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# from gradescope_utils.autograder_utils.decorators import weight
from hw4_utils import TestCase, run_tests
from hw4_mnist import MNISTModel, load_data


class Part2Tests(TestCase):
    def __init__(self, *args, **kwargs):
        self.model = MNISTModel()
        self.model.load()
        self.data = load_data()

        self.module_name = "hw4_part2"
        TestCase.__init__(self, *args, **kwargs)

#    @weight(0)
    def test_augment_defender_exists(self):
        """ AugmentDefender exists """

        self.assertHasAttribute(
            self.solution, "AugmentDefender"
        )

        import hw4_part1

        att = hw4_part1.PGDAttacker(
            model=self.model,
            num_steps=10,
            target=5,
            learning_rate_decay=0.9,
            learning_rate=2.0,
            step_mode="project",
            c=0.2
        )

        defender = self.solution.AugmentDefender(
            attacker=att,
            model=self.model,
            finetune=True,
            epochs=10
        )

        self.assertHasAttribute(defender, "defend")

#    @weight(0)
    def test_premadry_defender_exists(self):
        """ PreMadryDefender exists """

        self.assertHasAttribute(
            self.solution, "PreMadryDefender"
        )

        import hw4_part1

        att = hw4_part1.PGDAttacker(
            model=self.model,
            num_steps=10,
            target=5,
            learning_rate_decay=0.9,
            learning_rate=2.0,
            step_mode="project",
            c=0.2
        )

        defender = self.solution.AugmentDefender(
            attacker=att,
            model=self.model,
            finetune=True,
            epochs=10
        )

        self.assertHasAttribute(defender, "defend")


if __name__ == '__main__':
    run_tests(Part2Tests)
