import numpy as np
import operator
import pandas as pd
import os
import tensorflow as tf
from sklearn.utils import resample

############################## Data import ##############################
df = pd.read_csv(os.path.abspath('dataset.csv'),header=None)
X = df.iloc[:, [2,3,4,5]].values
y = df.iloc[:, 11].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y) #30% test set

X_upsampled, y_upsampled = resample(X_train[y_train == 1], y_train[y_train == 1], replace=True, n_samples=X_train[y_train == 0].shape[0], random_state=1)
X_train = np.vstack((X_train[y_train==0], X_upsampled))
y_train = np.hstack((y_train[y_train==0], y_upsampled))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

################################################################################

np.random.seed(1)

model = tf.keras.models.load_model('NN_classifier.h5')
model.load_weights('NN_classifier.h5')
model.summary()


# history_dict = history.history
# print(history_dict.keys())

import matplotlib.pyplot as plt

y_test_pred = model.predict_classes(X_test_std, verbose=0)

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
print('Test Accuracy score: %.3f' % (accuracy_score(y_test, y_test_pred)))
print('Test ROCAUC score: %.3f' % (roc_auc_score(y_test, y_test_pred)))

from sklearn.metrics import confusion_matrix
confmat = confusion_matrix(y_true=y_test, y_pred=y_test_pred)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.5)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.title('Test data confusion matrix')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.tight_layout()
plt.show()