import shap
import pandas as pd

def get_shap_explainer(model, X_train, model_type):
    if model_type == 'xgboost':
        return shap.TreeExplainer(model)
    elif model_type == 'mlp':
        return shap.Explainer(model.predict_proba, X_train)
    else:
        raise ValueError(f"Unsupported model type for SHAP: {model_type}")


def correlation_matrix(shap_values, feature_names):
    values = shap_values.values if hasattr(shap_values, "values") else shap_values
    if values.ndim == 3:
        values = values.mean(axis=2)
    return pd.DataFrame(values, columns=feature_names).corr().round(3)