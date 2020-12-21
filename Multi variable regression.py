import pandas as pd
import os
import numpy as np

############################## Data import ##############################
df = pd.read_csv(os.path.abspath('example_dataset_final.csv'),header=None)
df.drop(0, axis=1, inplace=True)
df.drop([0,1], inplace=True)
df.columns = ['feature1', 'feature2', 'feature3', 'feature4', 'target']
df.reset_index(inplace=True, drop=True)

mapping = {'area1':1, 'area2':2, 'area3':3, 'area4':4}
df['feature4'] = df['feature4'].map(mapping)

df = df.apply(pd.to_numeric)

X = df.iloc[:, :-1].values
y = df['target'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) #30% test set
########################################################################

from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

import matplotlib.pyplot as plt
plt.scatter(y_train_pred, y_train_pred-y_train, c='steelblue', marker='o', edgecolors='white', label='Training data')
plt.scatter(y_test_pred, y_test_pred-y_test, c='limegreen', marker='s', edgecolors='white', label='Test data')
plt.xlabel('Predicted values', fontsize=15)
plt.ylabel('Residuals', fontsize=15)
plt.legend(loc='upper left', fontsize=12)
plt.hlines(y=0, xmin=25000, xmax=200000, color='black', lw=2)
plt.xlim([25000, 200000])
plt.tight_layout()
plt.show()

from sklearn.metrics import r2_score
print('train R^2: %.3f, test R^2: %.3f' %(r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))

os.system("pause")