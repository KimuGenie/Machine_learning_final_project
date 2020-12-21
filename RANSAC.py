import pandas as pd
import os
import numpy as np

############################## Data import ##############################
df = pd.read_csv(os.path.abspath('example_dataset_final.csv'),header=None)
df.drop(0, axis=1, inplace=True)
df.drop([0,1], inplace=True)

X = df.iloc[:, 0].values.astype(np.float)
y = df.iloc[:, 4].values.astype(np.float)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) #30% test set

from sklearn.impute import SimpleImputer
imp = SimpleImputer()
X_train = imp.fit_transform(X_train.reshape(-1,1))
X_test = imp.fit_transform(X_test.reshape(-1,1))
#########################################################################

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
ransac = RANSACRegressor(LinearRegression(), max_trials=100, min_samples=10, loss='absolute_loss', residual_threshold=10000, random_state=0)
ransac.fit(X_train, y_train)

import matplotlib.pyplot as plt
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_x = np.arange(0, 180000, 10000)
line_y_ransac = ransac.predict(line_x[:, np.newaxis])
plt.scatter(X_train[inlier_mask], y_train[inlier_mask], c='steelblue', edgecolor='white', s=50, marker='o', label='Inliers')
plt.scatter(X_train[outlier_mask], y_train[outlier_mask], c='red', edgecolor='white', s=50, marker='s', label='Outliers')
plt.plot(line_x, line_y_ransac, color='black', lw=2)
plt.xlabel('feature1', fontsize=15)
plt.ylabel('target', fontsize=15)
plt.legend(loc='upper left', fontsize=12)
plt.tight_layout()
plt.show()

y_train_pred = ransac.predict(X_train)
y_test_pred = ransac.predict(X_test)

plt.scatter(X_test, y_test, c='limegreen', edgecolor='white', s=50, marker='o', label='Test data')
plt.plot(line_x, line_y_ransac, color='black', lw=2)
plt.xlabel('feature1', fontsize=15)
plt.ylabel('target', fontsize=15)
plt.legend(loc='upper left', fontsize=12)
plt.tight_layout()
plt.show()

from sklearn.metrics import r2_score
print('train R^2: %.3f, test R^2: %.3f' %(r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))

os.system("pause")