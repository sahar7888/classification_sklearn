import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/Iris.csv")

# print(data.head())
print(data['Species'].value_counts())

x = data.drop(['Id', 'Species'], axis=1)
y = data['Species']

x_train, x_test, y_train, y_test = train_test_split(x,y,)
