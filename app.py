import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

sc= joblib.load('save/sc.model')

df = pd.read_csv('data/sample.csv')
features = df.columns[0: -1]

classifiers = {
    "Isolation Forest": joblib.load('save/if.model'),
    "Local Outlier Factor": joblib.load('save/lof.model'),
    "Support Vector Machine": joblib.load('save/svm.model')
}

clf_names = classifiers.keys()


st.header("Anomaly Detection of Credit Card Fraud")
options = st.multiselect('Choose models', clf_names)