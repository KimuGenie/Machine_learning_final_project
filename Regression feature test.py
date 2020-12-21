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
########################################################################
# print(df.head())

import matplotlib.pyplot as plt

plt.figure(1)
df.plot.scatter(x='feature1', y='target', fontsize=12)
plt.xlabel('feature1', fontsize=12)
plt.ylabel('target', fontsize=12)
plt.tight_layout()

plt.figure(2)
df.plot.scatter(x='feature2', y='target', fontsize=12)
plt.xlabel('feature2', fontsize=12)
plt.ylabel('target', fontsize=12)
plt.tight_layout()

plt.figure(3)
df.plot.scatter(x='feature3', y='target', fontsize=12)
plt.xlabel('feature3', fontsize=12)
plt.ylabel('target', fontsize=12)
plt.tight_layout()

plt.figure(4)
df.plot.scatter(x='feature4', y='target', fontsize=12)
plt.xlabel('feature4', fontsize=12)
plt.ylabel('target', fontsize=12)
plt.tight_layout()

plt.show()