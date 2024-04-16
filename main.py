import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

"""Load Data"""
data = pd.read_csv("data/Iris.csv")

# print(data.head())
print(data['Species'].value_counts())
"""Data Pre-processing"""
x = data.drop(['Id', 'Species'], axis=1)
y = data['Species']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

"""Create the Model"""
model = LogisticRegression()

"""Training"""
model.fit(x_train,y_train)

""" Evaluation"""
y_pred = model.predict(x_test)

print(metrics.accuracy_score(y_test,y_pred))
"""Prediction"""
model.predict(np.array([[2,3,4,5]]))
