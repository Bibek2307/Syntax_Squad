import unittest
import requests
import json
import os
import sys
import time
import threading
import uvicorn

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.main import app

class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Start the API server in a separate thread"""
        cls.api_thread = threading.Thread(
            target=uvicorn.run,
            args=(app,),
            kwargs={"host": "127.0.0.1", "port": 8000, "log_level": "error"},
            daemon=True
        )
        cls.api_thread.start()
        time.sleep(1)  # Give the server time to start
        
    def test_predict_endpoint(self):
        """Test the predict endpoint"""
        # Create sample application data
        application = {
            "income": 50000,
            "credit_score": 720,
            "debt_to_income": 25,
            "loan_amount": 200000,
            "loan_term": 30,
            "employment_length": 5,
            "home_ownership": "OWN",
            "loan_purpose": "HOME"
        }
        
        # Make request to predict endpoint
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=application
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        
        # Parse response
        result = response.json()
        
        # Check that prediction is included
        self.assertIn("prediction", result)
        self.assertIn(result["prediction"], [0, 1])
        
        # Check that probability is included
        self.assertIn("probability", result)
        self.assertTrue(0 <= result["probability"] <= 1)
        
        # Check that explanation is included
        self.assertIn("explanation", result)
        self.assertIn("text", result["explanation"])
        
    def test_what_if_endpoint(self):
        """Test the what-if endpoint"""
        # Create sample application data
        application = {
            "income": 50000,
            "credit_score": 720,
            "debt_to_income": 25,
            "loan_amount": 200000,
            "loan_term": 30,
            "employment_length": 5,
            "home_ownership": "OWN",
            "loan_purpose": "HOME"
        }
        
        # Make request to what-if endpoint
        response = requests.post(
            "http://127.0.0.1:8000/what-if",
            json={
                "application": application,
                "feature": "income",
                "values": [30000, 40000, 50000, 60000, 70000]
            }
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        
        # Parse response
        result = response.json()
        
        # Check that results are included
        self.assertIn("results", result)
        self.assertEqual(len(result["results"]), 5)
        
        # Check that each result has value, prediction, and probability
        for item in result["results"]:
            self.assertIn("value", item)
            self.assertIn("prediction", item)
            self.assertIn("probability", item)

if __name__ == "__main__":
    unittest.main()