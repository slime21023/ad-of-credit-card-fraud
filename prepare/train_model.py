import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
df = pd.read_csv('../data/sample.csv')
df['Time'] = sc.fit_transform(df['Time'].values.reshape(-1, 1))
df['Amount'] = sc.fit_transform(df['Amount'].values.reshape(-1, 1))

features = df.columns[0: -1]
classifiers = {
    "Isolation Forest":IsolationForest(n_estimators=100),
    "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20, algorithm='auto', leaf_size=30, metric='minkowski',p=2, metric_params=None, novelty=True),
    "Support Vector Machine":OneClassSVM(kernel='rbf', degree=3, gamma=0.1,nu=0.05, max_iter=-1)  
}

clf_names = classifiers.keys()
for name in clf_names:
    classifiers[name].fit(df[features])

joblib.dump(sc, '../save/sc.model')
joblib.dump(classifiers["Isolation Forest"], '../save/if.model')
joblib.dump(classifiers["Local Outlier Factor"], '../save/lof.model')
joblib.dump(classifiers["Support Vector Machine"], '../save/svm.model')