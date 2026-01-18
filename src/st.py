import streamlit as st
import pandas as pd
from main import Classifier
import matplotlib.pyplot as plt
from shap_rf import run_shap_random_forest

st.title("Fraud Detection")
tab1, tab2 = st.tabs(["Model Overview", "Live Prediction"])

with tab1:
    st.header("Training Dataset")
    df = pd.read_csv("../data/raw/fraud_transactions.csv")
    st.dataframe(df)

    st.download_button(
        label="Download Training Dataset as CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='fraud_transactions.csv',
        mime='text/csv',
    )

    X_train, X_test, y_train, y_test = Classifier.data(df)
    rf, rfp = Classifier.RandForest(X_train, y_train)

    y_pred_train, y_prob_train = Classifier.prediction(model=rfp, X=X_train)
    y_pred_test, y_prob_test = Classifier.prediction(model=rfp, X=X_test)
    
    train_accuracy, train_f1, train_prec, train_rec, train_roc, train_cm = Classifier.metrics(y_train, y_pred_train, y_pred_train)
    test_accuracy, test_f1, test_prec, test_rec, test_roc, test_cm = Classifier.metrics(y_test, y_pred_test, y_prob_test)

    st.header("Important Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Training Metrics")
        st.metric(label="Train Accuracy", value=f"{train_accuracy*100:.2f}%")
        st.metric(label="Train F1", value=f"{train_f1*100:.2f}%")
        st.metric(label="Train Precision", value=f"{train_prec*100:.2f}%")
        st.metric(label="Train Recall", value=f"{train_rec*100:.2f}%")
        st.metric(label="Train ROC-AUC", value=f"{train_roc*100:.2f}%")
        st.write("Confusion Matrix:\n", train_cm)
    
    with col2:
        st.subheader("Testing Metrics")
        st.metric(label="Test Accuracy:",value= f"{test_accuracy*100:.2f}%")
        st.metric(label="Test F1:",value= f"{test_f1*100:.2f}%")
        st.metric(label="Test Precision:",value= f"{test_prec*100:.2f}%")
        st.metric(label="Test Recall:",value= f"{test_rec*100:.2f}%")
        st.metric(label="Test ROC-AUC:",value= f"{test_roc*100:.2f}%")
        st.write("Confusion Matrix:\n", test_cm)


    st.header("SHAP Explainability")
    dataset_choice = st.radio("Pick the dataset on which you want to run shap analysis:\n",["Training Dataset", "Testing Dataset"])
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

with tab2:
    data = st.file_uploader("Please upload your file for prediction", type='csv')
    if data:
        live_df = pd.read_csv(data)
        st.subheader("Provided Live Transactions")
        st.dataframe(live_df)

        X_test = Classifier.data(live_df)
        fraud_flag = Classifier.prediction(model= rfp, X=X_test)
        
        st.subheader("Live Predicted Fraud Transactions")
        live_df['fraud_flag'] = fraud_flag[0]
        st.dataframe(live_df)
        st.write("""Note, In fraud_flag column,
- '0' represents legitimate transaction
- '1' represents fraud transaction""")

        st.download_button(
        label="Download Predicted File as CSV",
        data=live_df.to_csv(index=False).encode('utf-8'),
        file_name='live_predicted_fraud_transactions.csv',
        mime='text/csv',
        )

        st.success("Prediction Successful")


