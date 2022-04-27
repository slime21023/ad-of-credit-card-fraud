import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sc= joblib.load('save/sc.model')
data = pd.read_csv('data/test_sample.csv')
features = data.columns[0: -1]

classifiers = {
    "Isolation Forest": joblib.load('save/if.model'),
    "Local Outlier Factor": joblib.load('save/lof.model'),
    "Support Vector Machine": joblib.load('save/svm.model')
}

clf_names = classifiers.keys()

def pred(opt, data):
    return list(map(lambda c: 0 if c == 1 else 1, classifiers[opt].predict(data[features])))

st.header("Anomaly Detection of Credit Card Fraud")

uploaded_file = st.file_uploader("Choose a CSV file")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data['Time'] = sc.fit_transform(data['Time'].values.reshape(-1, 1))
    data['Amount'] = sc.fit_transform(data['Amount'].values.reshape(-1, 1))
    st.write(data.head())

options = st.multiselect('Choose models', clf_names, ['Isolation Forest', 'Support Vector Machine'])
confusion_matrixs = {}
score_df = pd.DataFrame(columns=['model', 'tn', 'fp', 'fn', 'tp', 'recall', 'precision', 'f1-score'])
for opt in options:
    confusion_matrixs[opt] = confusion_matrix(data['Class'], pred(opt, data[features]), labels=[0, 1])
    tn, fp, fn, tp = confusion_matrixs[opt].ravel()
    recall = tp/(tp+fn)
    precision =  tp/(tp+fp)
    f1 = 2 * precision * recall / (recall + precision)
    score = pd.DataFrame([{ 'model': opt, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'recall': recall, 'precision':precision, 'f1-score': f1}])
    score_df = pd.concat([score_df, score], ignore_index= True)

st.dataframe(score_df)

length =  len(options)
fig, ax = plt.subplots(1, length, figsize=(10, 7))
for i in range(length):
    cmp = ConfusionMatrixDisplay(confusion_matrixs[options[i]], display_labels=[0, 1])
    if length == 1:
        cmp.plot(ax=ax)
    else:    
        cmp.plot(ax=ax[i])

st.pyplot(fig)