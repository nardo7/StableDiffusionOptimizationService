from experiments import InferenceExperiment, InferenceExperimentConfiguration
import logging
import unittest

class TestExperiment(unittest.TestCase):
    factors = ["factor1", "factor2"]
    levels = [[1, 2], [3, 4]]
    def test(self):
        configs = InferenceExperimentConfiguration()
        configs.factors = self.factors
        configs.levels = self.levels
        experiment_expected = [
            {
                "factor1": 1,
                "factor2": 3
            },
            {
                "factor1": 2,
                "factor2": 3
            },
            {
                "factor1": 1,
                "factor2": 4
            },
            {
                "factor1": 2,
                "factor2": 4
            }
        ]
        configs.check()
        exp = InferenceExperiment(logging.getLogger(), configs)
        print(exp.experiment_configs)
        for e in exp.experiment_configs:
            print(e)
            for k, v in e.items():
                self.assertEqual(v, experiment_expected[exp.experiment_configs.index(e)][k], f"{e[k]} != {experiment_expected[exp.experiment_configs.index(e)][k]}")

    def test_experiment_name(self):
        experiment = {
            "factor1": 1,
            "factor2": 3
        }

        repetition = 1
        configs = InferenceExperimentConfiguration()
        configs.factors = self.factors
        configs.levels = self.levels
        exp = InferenceExperiment(logging.getLogger(), configs)
        expected_name = "repetition_1__factor1_1__factor2_3"
        name = exp._generate_experiment_name(experiment, repetition=repetition)
        self.assertEqual(name, expected_name, f"{name} != {expected_name}")

        
if __name__ == '__main__':
    unittest.main()