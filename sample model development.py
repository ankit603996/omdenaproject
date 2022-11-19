# ML_model.py
import os

from sklearn import datasets
from xgboost import XGBClassifier
import pickle
import pandas as pd
import lime
import os
import lime.lime_tabular

data = pd.read_csv(r'Merged_dataset')

#Importing dataset from sklearn
X = data[[ 'Age', 'SystolicBP', 'DiastolicBP', 'BS']]
y = data['Risklabels']
#Create an XGB classifier
clf = XGBClassifier()
clf.fit(X, y)

# Export the ML model
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

# export explainer
explainer = lime.lime_tabular.LimeTabularExplainer(X.values, feature_names=X.columns.values.tolist(),
                                                  class_names=['0','1','2'], verbose=True, mode='classification')
import dill
with open('explainer', 'wb') as f:
    dill.dump(explainer, f)