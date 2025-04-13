import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import sys
import pickle

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

class LiverDiseaseModel:
    # In the __init__ method, add more detailed error handling
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_path = os.path.join(project_root, "models", "liver_model.joblib")
        self.scaler_path = os.path.join(project_root, "models", "liver_scaler.joblib")
        self.pkl_model_path = os.path.join(project_root, "models", "liver_disease_model.pkl")
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        print(f"Looking for model at: {self.pkl_model_path}")
        
        # Try to load model and scaler in this order:
        # 1. First try the .pkl file
        # 2. Then try the .joblib files
        if os.path.exists(self.pkl_model_path):
            try:
                print(f"Loading model from {self.pkl_model_path}")
                with open(self.pkl_model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    
                # Check if the loaded data is a dictionary containing both model and scaler
                if isinstance(model_data, dict):
                    self.model = model_data.get('model')
                    self.scaler = model_data.get('scaler')
                    print("Successfully loaded model and scaler from .pkl file")
                else:
                    # If it's just the model
                    self.model = model_data
                    print("Loaded model from .pkl file, but no scaler found")
                    
                    # Try to load scaler separately if it exists
                    if os.path.exists(self.scaler_path):
                        self.scaler = joblib.load(self.scaler_path)
                        print("Loaded scaler from .joblib file")
                    else:
                        # Create a default scaler if none exists
                        print("No scaler found, creating a default StandardScaler")
                        self.scaler = StandardScaler()
            except Exception as e:
                print(f"Error loading model from .pkl file: {str(e)}")
                import traceback
                print(traceback.format_exc())
        else:
            print(f"Model file not found at: {self.pkl_model_path}")
    
    def train(self, X, y):
        """Train the model on the provided data"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and fit the scaler
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Initialize and train the model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Save the model and scaler in both formats
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        
        # Also save as .pkl for compatibility
        with open(self.pkl_model_path, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
        
        # Evaluate on test set
        X_test_scaled = self.scaler.transform(X_test)
        test_score = self.model.score(X_test_scaled, y_test)
        return test_score
    
    def predict(self, features):
        """Make a prediction for the given features"""
        if self.model is None:
            raise ValueError(f"Model not loaded. Please ensure model file exists at {self.pkl_model_path} and is valid.")
        
        if self.scaler is None:
            print("Warning: No scaler found. Using raw features without scaling.")
        
        # Convert features to DataFrame
        feature_names = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 
                        'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 
                        'Aspartate_Aminotransferase', 'Total_Protiens', 
                        'Albumin', 'Albumin_and_Globulin_Ratio']
        
        # Create a DataFrame with the features in the correct order
        df = pd.DataFrame([features], columns=feature_names)
        
        # Scale the features if scaler is available
        if self.scaler is not None:
            try:
                X_scaled = self.scaler.transform(df)
            except Exception as e:
                print(f"Error scaling features: {str(e)}. Using raw features.")
                X_scaled = df.values
        else:
            X_scaled = df.values
        
        # Make prediction
        try:
            prediction = bool(self.model.predict(X_scaled)[0])
            probability = float(self.model.predict_proba(X_scaled)[0][1])
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise ValueError(f"Error making prediction: {str(e)}")
        
        return {
            "prediction": prediction,
            "probability": probability
        }
        
    def get_feature_importance(self):
        """Return feature importance if available"""
        if self.model is None:
            return None
            
        try:
            # Get feature importance from the model
            if hasattr(self.model, 'feature_importances_'):
                return self.model.feature_importances_.tolist()
            return None
        except:
            return None