import unittest
from collections import Counter

import pandas as pd
from math import isclose

from neural_network.task1.GainRation import GainRation


class TestGainRation(unittest.TestCase):
    __df = pd.read_csv('./resources/test_dataset.csv')

    def test_calculate_entropy(self):
        test_counter = Counter({'No': 5, 'Yes': 5})
        entropy = GainRation._calculate_entropy(test_counter, 10)
        expected_entropy = 1.0
        self.assertTrue(isclose(entropy, expected_entropy, abs_tol=1e-5))

    def test_inform_entropy_t(self):
        entropy_t = GainRation._inform_entropy_t(self.__df, 'Job')
        expected_entropy_t = 0.8812
        self.assertTrue(isclose(entropy_t, expected_entropy_t, abs_tol=1e-4))

    def test_inform_entropy_a_t(self):
        entropy_a_t = GainRation._inform_entropy_a_t(self.__df, 'Pratical Knowlage', 'Job')
        expected_entropy_a_t = 0.63645
        self.assertTrue(isclose(entropy_a_t, expected_entropy_a_t, abs_tol=1e-5))

    def test_gain_ration(self):
        gain_ration = GainRation.gain_ration(self.__df, 'CGPT', 'Job')
        expected_gain_ration = 0.3658
        self.assertTrue(isclose(gain_ration, expected_gain_ration, abs_tol=1e-4))
