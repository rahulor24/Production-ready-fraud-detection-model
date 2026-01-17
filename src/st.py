import streamlit as st
import pandas as pd
from main import RandForest
import matplotlib.pyplot as plt
from shap_rf import run_shap_random_forest

st.title("Fraud Detection")

file = st.file_uploader("Please upload your file", type='csv')
if file:
    df = pd.read_csv(file)
    st.dataframe(df)

    train_f1, test_f1, cm, rf, X_train, X_test = RandForest(df)

    st.write("Train F1:", train_f1)
    st.write("\nTest F1:", test_f1)
    st.write("\nConfusion Matrix:\n", cm)


    st.header("Random Forest SHAP Explainability")
    dataset_choice = st.radio("Pick the dataset on which you want to run shap:\n",["Training Dataset", "Testing Dataset"])
    X_input = X_train if dataset_choice == "Training Dataset" else X_test

    n_display = st.slider("Number of best features to display:",5,20,10)

    if st.button("Run SHAP Analysis"):
        
        shap_values, X_shap, fig_bar, fig_beeswarm = run_shap_random_forest(
        rf,
        X=X_input,
        sample_size=1000,
        max_display=n_display
        )

        st.subheader("Global Feature Importance (Bar)")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Detailed Feature Impact (Beeswarm)")
        st.plotly_chart(fig_beeswarm, use_container_width=True)

        st.success("SHAP analysis completed")

