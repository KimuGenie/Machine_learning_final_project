import numpy as np
import operator
import pandas as pd
import os
import tensorflow as tf
from sklearn.utils import resample

############################## Data import ##############################
df = pd.read_csv(os.path.abspath('dataset.csv'),header=None)
X = df.iloc[:, :11].values
y = df.iloc[:, 11].values
X_org = X #upsampling 하지 않을 원본 데이터
y_org = y

X_upsampled, y_upsampled = resample(X[y == 1], y[y == 1], replace=True, n_samples=X[y == 0].shape[0], random_state=1)
X = np.vstack((X[y==0], X_upsampled))
y = np.hstack((y[y==0], y_upsampled))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y) #30% test set
X_train_org, X_test_org, y_train_org, y_test_org = train_test_split(X_org, y_org, test_size=0.3, random_state=1, stratify=y_org)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_org_std = sc.transform(X_org)

y_train_onehot = tf.keras.utils.to_categorical(y_train)
################################################################################

np.random.seed(1)

model = tf.keras.models.Sequential()

model.add(
    tf.keras.layers.Dense(
        units=6,
        input_dim=X_train_std.shape[1],
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='tanh'
    )
)

model.add(
    tf.keras.layers.Dense(
        units=6,
        input_dim=6,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='tanh'
    )
)

model.add(
    tf.keras.layers.Dense(
        units=6,
        input_dim=6,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='tanh'
    )
)

model.add(
    tf.keras.layers.Dense(
        units=y_train_onehot.shape[1],
        input_dim=6,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='softmax'
    )
)

# model.summary()

sgd_optimizer = tf.keras.optimizers.SGD(lr=0.003, decay=1e-7, momentum=.9)

model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy')

history = model.fit(X_train_std, y_train_onehot, batch_size=50, epochs=50, verbose=1, validation_split=0.1)

y_train_pred = model.predict_classes(X_train_std, verbose=0)
correct_preds = np.sum(y_train == y_train_pred, axis=0)
train_acc = correct_preds/y_train.shape[0]
print('train accuracy:  %0.3f' %(train_acc*100))
y_test_pred = model.predict_classes(X_test_std, verbose=0)
correct_preds = np.sum(y_test == y_test_pred, axis=0)
test_acc = correct_preds/y_test.shape[0]
print('test accuracy:  %0.3f' %(test_acc*100))
y_org_pred = model.predict_classes(X_org_std, verbose=0)
correct_preds = np.sum(y_org == y_org_pred, axis=0)
org_acc = correct_preds/y_org.shape[0]
print('original accuracy:  %0.3f' %(org_acc*100))

print('a')