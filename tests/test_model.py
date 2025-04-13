import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.loan_model import LoanApprovalModel

class TestLoanModel(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.model = LoanApprovalModel()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            "income": [20000, 50000, 80000, 30000],
            "credit_score": [580, 720, 800, 650],
            "debt_to_income": [40, 25, 15, 30],
            "loan_amount": [150000, 200000, 300000, 180000],
            "loan_term": [30, 30, 15, 30],
            "employment_length": [2, 5, 10, 3],
            "home_ownership": ["RENT", "OWN", "OWN", "RENT"],
            "loan_purpose": ["HOME", "HOME", "HOME", "HOME"]
        })
        
    def test_predict(self):
        """Test prediction functionality"""
        predictions = self.model.predict(self.sample_data)
        
        # Check that predictions are binary (0 or 1)
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
        
        # Check that higher income and credit score leads to approval
        self.assertEqual(predictions[2], 1, "High income and credit score should lead to approval")
        
    def test_predict_proba(self):
        """Test probability prediction functionality"""
        probabilities = self.model.predict_proba(self.sample_data)
        
        # Check that probabilities are between 0 and 1
        self.assertTrue(all(0 <= prob[1] <= 1 for prob in probabilities))
        
        # Check that probabilities are higher for better applications
        self.assertGreater(probabilities[1][1], probabilities[0][1], 
                          "Better application should have higher approval probability")
        
    def test_feature_importance(self):
        """Test feature importance functionality"""
        importance = self.model.feature_importance()
        
        # Check that feature importance is returned for all features
        self.assertEqual(len(importance), len(self.sample_data.columns))
        
        # Check that importance values sum to approximately 1
        self.assertAlmostEqual(sum(importance.values()), 1.0, places=5)

if __name__ == "__main__":
    unittest.main()