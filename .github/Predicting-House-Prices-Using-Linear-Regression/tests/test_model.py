import unittest
from src.model import LinearRegressionModel
import pandas as pd

class TestLinearRegressionModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load sample data for testing
        cls.data = pd.read_csv('data/processed/processed_data.csv')
        cls.features = cls.data.drop('MEDV', axis=1)  # Assuming 'MEDV' is the target variable
        cls.target = cls.data['MEDV']
        cls.model = LinearRegressionModel()

    def test_model_training(self):
        self.model.train(self.features, self.target)
        self.assertIsNotNone(self.model.coefficients, "Model coefficients should not be None after training.")

    def test_model_prediction(self):
        predictions = self.model.predict(self.features)
        self.assertEqual(len(predictions), len(self.target), "Number of predictions should match number of target values.")

    def test_model_coefficients(self):
        self.model.train(self.features, self.target)
        self.assertGreater(len(self.model.coefficients), 0, "Model should have non-zero coefficients after training.")

if __name__ == '__main__':
    unittest.main()