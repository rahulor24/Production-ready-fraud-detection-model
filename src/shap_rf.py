import shap
import pandas as pd
import numpy as np
import plotly.express as px
from preprocessor import Preprocessor


def run_shap_random_forest(
    rf,
    X,
    sample_size=1000,
    random_state=42,
    max_display=20
):
    """
    Computes SHAP values for a trained Random Forest pipeline.

    Parameters
    ----------
    best_model_rf : sklearn Pipeline
        Trained pipeline with steps ['preprocessor', 'classifier']
    X : pandas DataFrame
        Input data (train or test)
    sample_size : int
        Number of rows to sample for SHAP (for speed)
    random_state : int
        Random seed
    max_display : int
        Number of top features to display

    Returns
    -------
    shap_values_class1 : np.ndarray
        SHAP values for positive class
    X_shap_df : pandas DataFrame
        Transformed feature matrix with feature names
    """

    # -----------------------------
    # Extract components
    # -----------------------------
    preprocessor = Preprocessor()
    rf_model = rf

    # -----------------------------
    # Transform data
    # -----------------------------
    X_transformed = preprocessor.fit_transform(X)
    feature_names = preprocessor.named_steps["transformer"].get_feature_names_out()

    X_df = pd.DataFrame(X_transformed, columns=feature_names)

    # -----------------------------
    # Sample for SHAP (important)
    # -----------------------------
    if len(X_df) > sample_size:
        X_shap_df = X_df.sample(sample_size, random_state=random_state)
    else:
        X_shap_df = X_df.copy()

    X_shap_array = X_shap_df.values  # force numpy

    # -----------------------------
    # SHAP Explainer
    # -----------------------------
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_shap_array)

    # -----------------------------
    # Handle SHAP output safely
    # -----------------------------
    # Binary classification → class 1
    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]
    else:
        # SHAP >= 0.43 sometimes returns (n, f, 2)
        shap_values_class1 = shap_values[:, :, 1]

    # -----------------------------
    # Sanity check (prevents crash)
    # -----------------------------
    assert shap_values_class1.shape[1] == X_shap_array.shape[1], \
        "SHAP values and feature matrix shape mismatch"

    # -----------------------------
    # GLOBAL FEATURE IMPORTANCE (Plotly BAR)
    # -----------------------------
    mean_abs_shap = np.abs(shap_values_class1).mean(axis=0)

    shap_importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False).head(max_display)

    fig_bar = px.bar(
        shap_importance_df,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        title="Global Feature Importance (Mean |SHAP|)"
    )

    fig_bar.update_layout(yaxis=dict(autorange="reversed"))

    # -----------------------------
    # DETAILED BEESWARM (Plotly)
    # -----------------------------
    shap_long_df = pd.DataFrame(
        shap_values_class1[:, shap_importance_df.index],
        columns=shap_importance_df["feature"]
    )

    shap_long_df["row_id"] = np.arange(len(shap_long_df))

    shap_melted = shap_long_df.melt(
        id_vars="row_id",
        var_name="feature",
        value_name="shap_value"
    )

    fig_beeswarm = px.strip(
        shap_melted,
        x="shap_value",
        y="feature",
        orientation="h",
        title="SHAP Beeswarm Plot (Feature Impact)"
    )

    fig_beeswarm.update_traces(opacity=0.6, jitter=0.4)

    return shap_values_class1, X_shap_df, fig_bar, fig_beeswarm

