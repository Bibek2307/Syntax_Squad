import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import shap

def plot_feature_importance(model, feature_names=None):
    """Plot feature importance for tree-based models"""
    if feature_names is None and hasattr(model, 'feature_names'):
        feature_names = model.feature_names
        
    if hasattr(model, 'model'):
        importances = model.model.feature_importances_
    else:
        importances = model.feature_importances_
        
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance')
    plt.bar(range(len(indices)), importances[indices], align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    return plt
    
def plot_shap_summary(explainer, X):
    """Plot SHAP summary plot"""
    shap_values = explainer.explainer.shap_values(X)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # For binary classification
        
    return shap.summary_plot(shap_values, X)
    
def plot_what_if(model, instance, feature_to_vary, range_values, feature_names=None):
    """Plot what-if analysis for a single feature"""
    instance_df = instance.copy()
    predictions = []
    
    for value in range_values:
        instance_df[feature_to_vary] = value
        if hasattr(model, 'predict_proba'):
            pred = model.predict_proba(instance_df)[0][1]
        else:
            pred = model.model.predict_proba(instance_df)[0][1]
        predictions.append(pred)
        
    plt.figure(figsize=(10, 6))
    plt.plot(range_values, predictions)
    plt.xlabel(feature_to_vary)
    plt.ylabel('Probability of Approval')
    plt.title(f'What-If Analysis: Impact of {feature_to_vary} on Loan Approval')
    plt.grid(True)
    return plt
    
def plot_comparative_analysis(model, instance_a, instance_b, explainer, feature_names=None):
    """Compare two instances and their explanations"""
    # Get predictions
    if hasattr(model, 'predict_proba'):
        pred_a = model.predict_proba(instance_a)[0][1]
        pred_b = model.predict_proba(instance_b)[0][1]
    else:
        pred_a = model.model.predict_proba(instance_a)[0][1]
        pred_b = model.model.predict_proba(instance_b)[0][1]
        
    # Get explanations
    shap_a = explainer.explain_instance(instance_a)
    shap_b = explainer.explain_instance(instance_b)
    
    # Create comparison dataframe
    comparison = pd.DataFrame({
        'Feature': feature_names,
        'Instance A': instance_a.values[0],
        'Instance B': instance_b.values[0],
        'SHAP A': shap_a[0],
        'SHAP B': shap_b[0]
    })
    
    # Calculate difference in SHAP values
    comparison['SHAP Diff'] = comparison['SHAP B'] - comparison['SHAP A']
    
    # Sort by absolute difference
    comparison = comparison.iloc[np.argsort(np.abs(comparison['SHAP Diff']))[::-1]]
    
    # Plot top differences
    plt.figure(figsize=(12, 8))
    sns.barplot(x='SHAP Diff', y='Feature', data=comparison.head(10))
    plt.title('Top Features Explaining the Difference in Predictions')
    plt.tight_layout()
    return plt