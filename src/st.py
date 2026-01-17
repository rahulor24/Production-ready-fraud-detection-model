import streamlit as st
import pandas as pd
import openpyxl
from main import RandForest

st.title("Fraud Detection")

file = st.file_uploader("Please upload your file", type= 'xlsx')
if file:
    df = pd.read_excel(file)
    st.dataframe(df)

    train_f1, test_f1, cm = RandForest(df)
    
    st.write("Train F1:", train_f1)
    st.write("\nTest F1:", test_f1)
    st.write("\nConfusion Matrix:\n", cm)

