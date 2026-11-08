import unittest
from types import SimpleNamespace as NS
from xvla_rlinf.contracts import validate_algorithm

class AlgorithmTests(unittest.TestCase):
    def test_only_supported_training_pairs_are_accepted(self):
        for pair in [('gae','actor_critic'),('grpo','actor')]:
            validate_algorithm(NS(runner=NS(only_eval=False),algorithm=NS(adv_type=pair[0],loss_type=pair[1])))
        for pair in [('gae','actor'),('grpo','actor_critic'),('sac','sac'),('iql','iql'),('opd','opd'),('gae','embodied_dagger')]:
            with self.subTest(pair=pair),self.assertRaises(NotImplementedError):
                validate_algorithm(NS(runner=NS(only_eval=False),algorithm=NS(adv_type=pair[0],loss_type=pair[1])))

    def test_evaluation_does_not_require_training_fields(self):
        validate_algorithm(NS(runner=NS(only_eval=True)))
