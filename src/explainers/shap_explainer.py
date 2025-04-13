import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ShapExplainer:
    def __init__(self, model, feature_names=None):
        self.model = model
        self.explainer = None
        self.feature_names = feature_names
        
    def fit(self, X_background):
        """Fit the explainer with background data"""
        # Transform data if preprocessor exists
        if hasattr(self.model, 'preprocessor') and self.model.preprocessor is not None:
            X_processed = self.model.preprocessor.transform(X_background)
            # Convert to dense array if sparse
            if hasattr(X_processed, "toarray"):
                X_processed = X_processed.toarray()
        else:
            X_processed = X_background
            
        # Create explainer
        if hasattr(self.model, 'model'):
            self.explainer = shap.TreeExplainer(self.model.model)
        else:
            self.explainer = shap.TreeExplainer(self.model)
            
    def explain_instance(self, instance):
        """Generate SHAP values for a single instance"""
        # Transform instance if preprocessor exists
        if hasattr(self.model, 'preprocessor') and self.model.preprocessor is not None:
            instance_processed = self.model.preprocessor.transform(instance)
            # Convert to dense array if sparse
            if hasattr(instance_processed, "toarray"):
                instance_processed = instance_processed.toarray()
        else:
            instance_processed = instance
            
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(instance_processed)
        
        # For classification models, shap_values might be a list of arrays
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Assuming binary classification
            
        return shap_values
        
    def generate_explanation(self, instance, original_features=None):
        """Generate human-readable explanation"""
        shap_values = self.explain_instance(instance)
        
        # Get feature names
        if self.feature_names is None and hasattr(self.model, 'feature_names'):
            self.feature_names = self.model.feature_names
            
        # If we have the original feature names and values
        if original_features is not None:
            # Sort features by absolute SHAP value
            feature_importance = [(name, abs(shap_values[0][i])) 
                                 for i, name in enumerate(self.feature_names)]
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            # Generate explanation text
            explanation = []
            for feature, importance in feature_importance[:5]:  # Top 5 features
                if importance > 0.01:  # Only include significant features
                    value = original_features[feature].values[0]
                    direction = "increased" if shap_values[0][self.feature_names.index(feature)] > 0 else "decreased"
                    explanation.append(f"{feature} = {value} {direction} the likelihood")
                    
            return " and ".join(explanation)
        
        return "Explanation requires original feature values"
        
    def plot_force(self, instance, matplotlib=True):
        """Generate force plot for an instance"""
        shap_values = self.explain_instance(instance)
        
        if matplotlib:
            shap.force_plot(self.explainer.expected_value, 
                           shap_values, 
                           instance, 
                           feature_names=self.feature_names,
                           matplotlib=True)
        else:
            return shap.force_plot(self.explainer.expected_value, 
                                  shap_values, 
                                  instance, 
                                  feature_names=self.feature_names)