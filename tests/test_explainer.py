import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.loan_model import LoanApprovalModel
from src.explainers.shap_explainer import ShapExplainer

class TestShapExplainer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.model = LoanApprovalModel()
        self.explainer = ShapExplainer(self.model)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            "income": [50000],
            "credit_score": [720],
            "debt_to_income": [25],
            "loan_amount": [200000],
            "loan_term": [30],
            "employment_length": [5],
            "home_ownership": ["OWN"],
            "loan_purpose": ["HOME"]
        })
        
    def test_generate_explanation(self):
        """Test explanation generation"""
        explanation = self.explainer.generate_explanation(self.sample_data)
        
        # Check that explanation is a string
        self.assertIsInstance(explanation, str)
        
        # Check that explanation contains key features
        self.assertTrue(any(feature in explanation.lower() for feature in 
                           ["income", "credit score", "debt", "loan"]))
        
    def test_get_feature_importance(self):
        """Test feature importance calculation"""
        importance = self.explainer.get_feature_importance(self.sample_data)
        
        # Check that importance is a list of dictionaries
        self.assertIsInstance(importance, list)
        self.assertIsInstance(importance[0], dict)
        
        # Check that each dictionary has feature and importance keys
        self.assertTrue(all("feature" in item and "importance" in item for item in importance))
        
        # Check that importance values are between -1 and 1
        self.assertTrue(all(-1 <= item["importance"] <= 1 for item in importance))
        
    def test_generate_plot(self):
        """Test plot generation"""
        plot_data = self.explainer.generate_plot(self.sample_data)
        
        # Check that plot data is returned
        self.assertIsNotNone(plot_data)

if __name__ == "__main__":
    unittest.main()