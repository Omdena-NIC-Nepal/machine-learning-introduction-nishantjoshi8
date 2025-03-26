import unittest
import pandas as pd
from src.evaluation import calculate_mse, calculate_r_squared

class TestEvaluation(unittest.TestCase):

    def setUp(self):
        self.y_true = pd.Series([3, -0.5, 2, 7])
        self.y_pred = pd.Series([2.5, 0.0, 2, 8])

    def test_calculate_mse(self):
        expected_mse = ((self.y_true - self.y_pred) ** 2).mean()
        self.assertAlmostEqual(calculate_mse(self.y_true, self.y_pred), expected_mse)

    def test_calculate_r_squared(self):
        ss_total = ((self.y_true - self.y_true.mean()) ** 2).sum()
        ss_residual = ((self.y_true - self.y_pred) ** 2).sum()
        expected_r_squared = 1 - (ss_residual / ss_total)
        self.assertAlmostEqual(calculate_r_squared(self.y_true, self.y_pred), expected_r_squared)

if __name__ == '__main__':
    unittest.main()