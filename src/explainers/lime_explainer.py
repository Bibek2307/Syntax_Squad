import lime
import lime.lime_tabular
import numpy as np
import pandas as pd

class LimeExplainer:
    def __init__(self, model, feature_names=None, class_names=None):
        self.model = model
        self.explainer = None
        self.feature_names = feature_names
        self.class_names = class_names or ['Denied', 'Approved']
        
    def fit(self, X_background):
        """Fit the explainer with background data"""
        # Get feature names
        if self.feature_names is None and hasattr(self.model, 'feature_names'):
            self.feature_names = self.model.feature_names
            
        # Create prediction function
        def predict_fn(x):
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(pd.DataFrame(x, columns=self.feature_names))
            else:
                return self.model.model.predict_proba(x)
                
        # Create explainer
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            X_background.values,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode='classification'
        )
        
    def explain_instance(self, instance, num_features=5):
        """Generate LIME explanation for a single instance"""
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
            
        # Define prediction function
        def predict_fn(x):
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(pd.DataFrame(x, columns=self.feature_names))
            else:
                return self.model.model.predict_proba(x)
                
        # Generate explanation
        explanation = self.explainer.explain_instance(
            instance.values[0], 
            predict_fn,
            num_features=num_features
        )
        
        return explanation
        
    def generate_explanation(self, instance, num_features=5):
        """Generate human-readable explanation"""
        explanation = self.explain_instance(instance, num_features)
        
        # Get the explanation as a list of tuples (feature, weight)
        features_weights = explanation.as_list()
        
        # Generate text explanation
        text_explanation = []
        for feature, weight in features_weights:
            direction = "increased" if weight > 0 else "decreased"
            text_explanation.append(f"{feature} {direction} the likelihood")
            
        return " and ".join(text_explanation)
        
    def plot_explanation(self, instance, num_features=5):
        """Plot the explanation"""
        explanation = self.explain_instance(instance, num_features)
        return explanation.as_pyplot_figure()