import pandas as pd
import numpy as np
import pickle
import os
import sys
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from typing import List

logger = logging.getLogger(__name__)

class DiabetesModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_metrics = None
        
        # Get the project root directory
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Set paths for model files
        self.model_path = os.path.join(self.project_root, 'models', 'diabetes_model.pkl')
        self.feature_names_path = os.path.join(self.project_root, 'models', 'diabetes_feature_names.pkl')
        self.model_metrics_path = os.path.join(self.project_root, 'models', 'diabetes_model_metrics.pkl')
        
        # Default feature names if not loaded from file
        self.default_feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        
        # Initialize feature names first
        self.feature_names = self.default_feature_names
        
        # Load the model and related files
        self.load_model()

    def load_model(self):
        """Load the trained model and related files from disk."""
        try:
            # Try to load feature names first
            if os.path.exists(self.feature_names_path):
                try:
                    with open(self.feature_names_path, 'rb') as f:
                        self.feature_names = pickle.load(f, encoding='latin1')
                        logger.info("Feature names loaded successfully")
                except Exception as e:
                    logger.warning(f"Error loading feature names: {str(e)}. Using defaults.")
                    self.feature_names = self.default_feature_names
            else:
                logger.warning("Feature names file not found, using defaults")
                self.feature_names = self.default_feature_names
        
            # Try to load the model
            if os.path.exists(self.model_path):
                try:
                    with open(self.model_path, 'rb') as f:
                        model_data = pickle.load(f, encoding='latin1')
                        if isinstance(model_data, dict):
                            self.model = model_data.get('model')
                            self.scaler = model_data.get('scaler')
                            if self.model is None or self.scaler is None:
                                raise ValueError("Model or scaler missing from loaded data")
                        else:
                            self.model = model_data
                            # Create a new scaler if not found in model data
                            self.scaler = StandardScaler()
                            logger.warning("Model loaded but scaler not found. Creating new scaler.")
                    logger.info("Model loaded successfully")
                except Exception as e:
                    logger.error(f"Error loading model: {str(e)}")
                    raise ValueError(f"Failed to load diabetes model: {str(e)}")
            else:
                logger.error("Model file not found.")
                raise FileNotFoundError(f"Diabetes model file not found at {self.model_path}")
        
            # Try to load model metrics
            if os.path.exists(self.model_metrics_path):
                try:
                    with open(self.model_metrics_path, 'rb') as f:
                        self.model_metrics = pickle.load(f, encoding='latin1')
                        logger.info("Model metrics loaded successfully")
                except Exception as e:
                    logger.warning(f"Error loading model metrics: {str(e)}")
                    self.model_metrics = None
            else:
                logger.warning("Model metrics file not found")
                self.model_metrics = None
        except Exception as e:
            logger.error(f"Error in load_model: {str(e)}")
            raise ValueError(f"Failed to load diabetes model: {str(e)}")
    
    # Remove the _create_dummy_model method entirely
    def _create_dummy_model(self):
        """Create a dummy model for testing purposes."""
        try:
            logger.warning("Creating dummy model")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()
            
            # Create dummy data to fit the scaler and model
            dummy_data = pd.DataFrame(np.random.randn(100, len(self.feature_names)), 
                                    columns=self.feature_names)
            self.scaler.fit(dummy_data)
            
            # Fit the model with dummy data
            dummy_target = np.random.randint(0, 2, 100)
            self.model.fit(dummy_data, dummy_target)
            logger.info("Dummy model created successfully")
        except Exception as e:
            logger.error(f"Error creating dummy model: {str(e)}")
            raise

    def save_model(self):
        """Save the model and scaler together in one file."""
        try:
            # Create a dictionary containing both model and scaler
            model_data = {
                'model': self.model,
                'scaler': self.scaler
            }
            
            # Save to file
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info("Model and scaler saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def predict(self, features):
        """Make a prediction using the trained model."""
        try:
            if self.model is None:
                raise ValueError("Model not loaded. Please ensure model file exists and is valid.")
            
            print(f"Input features for diabetes prediction: {features}")
            
            # Convert string inputs to appropriate numeric types
            processed_features = {}
            for key, value in features.items():
                try:
                    processed_features[key] = float(value)
                except (ValueError, TypeError):
                    # Handle conversion errors
                    raise ValueError(f"Invalid value for feature {key}: {value}. Expected numeric value.")
            
            # Create DataFrame with processed values
            X = pd.DataFrame([processed_features])
            
            # Ensure all required columns are present
            required_columns = [
                'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
            ]
            
            for col in required_columns:
                if col not in X.columns:
                    raise ValueError(f"Missing required feature: {col}")
            
            # Ensure columns are in the correct order
            X = X[required_columns]
            
            # Convert all data to float64 to ensure compatibility
            X = X.astype(float)
            
            # Scale features if scaler is available
            if hasattr(self, 'scaler') and self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X.values
            
            # Make prediction
            prediction = bool(self.model.predict(X_scaled)[0])
            
            # Get probability - handle different model types
            if hasattr(self.model, 'predict_proba'):
                # For models that provide probability
                proba = self.model.predict_proba(X_scaled)[0]
                # Make sure we get the probability for the positive class (index 1)
                probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
            else:
                # For models that don't provide probability
                probability = 0.5 + (float(self.model.decision_function(X_scaled)[0]) / 10)
                probability = max(0, min(1, probability))  # Clamp between 0 and 1
            
            return {
                "prediction": prediction,
                "probability": probability
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise ValueError(f"Error during prediction: {str(e)}")

    def get_feature_importance(self) -> List[float]:
        """Get the feature importance scores as a list of floats."""
        try:
            if hasattr(self.model, 'feature_importances_'):
                # Convert feature importances to a list of floats
                importances = [float(x) for x in self.model.feature_importances_]
                # Ensure we have the same number of importances as features
                if len(importances) == len(self.feature_names):
                    return importances
            # If we can't get valid feature importances, return None
            logger.warning("Could not get valid feature importances")
            return None
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return None

    def get_model_metrics(self):
        """Get the model metrics."""
        return self.model_metrics if self.model_metrics else None

    def train_model(self, X, y):
        """Train the model with the given data."""
        try:
            logger.info("Starting model training...")
            
            # Initialize the scaler and scale the features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Initialize and train the model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.model.fit(X_scaled, y)
            
            # Calculate and store model metrics
            train_score = self.model.score(X_scaled, y)
            feature_importance = self.model.feature_importances_
            
            self.model_metrics = {
                'train_score': train_score,
                'feature_importance': feature_importance.tolist()
            }
            
            # Save the model, scaler, and metrics
            self.save_model()
            self.save_metrics()
            self.save_feature_names()
            
            logger.info(f"Model trained successfully. Training score: {train_score:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Error in train_model: {str(e)}")
            raise

    def save_metrics(self):
        """Save model metrics to file."""
        try:
            with open(self.model_metrics_path, 'wb') as f:
                pickle.dump(self.model_metrics, f)
            logger.info("Model metrics saved successfully")
        except Exception as e:
            logger.error(f"Error saving model metrics: {str(e)}")
            raise

    def save_feature_names(self):
        """Save feature names to file."""
        try:
            with open(self.feature_names_path, 'wb') as f:
                pickle.dump(self.feature_names, f)
            logger.info("Feature names saved successfully")
        except Exception as e:
            logger.error(f"Error saving feature names: {str(e)}")
            raise

def train_model():
    """Train and save the diabetes prediction model"""
    try:
        model = DiabetesModel()
        
        # Get absolute paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        data_file = os.path.join(project_root, "data", "diabetes.csv")
        model_dir = os.path.join(project_root, 'models')
        
        print(f"Loading data from: {data_file}")
        print(f"Model will be saved to: {model_dir}")
        
        # Ensure data file exists
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found at {data_file}")
        
        # Create models directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Load data
        print("Loading and preparing data...")
        data = pd.read_csv(data_file)
        
        # Select features and target
        X = data[model.feature_names]
        y = data['Outcome']
        
        # Train the model
        print("Training model...")
        model.train_model(X, y)
        print("Model trained and saved successfully")
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    train_model()