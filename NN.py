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

model = tf.keras.models.Sequential()

model.add(
    tf.keras.layers.Dense(
        units=6,
        input_dim=X_train_std.shape[1],
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='relu'
    )
)

model.add(
    tf.keras.layers.Dense(
        units=6,
        input_dim=6,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='relu'
    )
)

#binary classification을 위해 마지막 레이어는 유닛 하나에 activation function으로 sigmoid function을 사용함
model.add(
    tf.keras.layers.Dense(
        units=1,
        input_dim=6,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='sigmoid'
    )
)

# model.summary()

sgd_optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-7, momentum=.9)

model.compile(optimizer=sgd_optimizer, loss='binary_crossentropy', metrics=['accuracy']) #0 또는 1로 분류하기 때문에 binary_crossentropy를 loss함수로 선정

history = model.fit(X_train_std, y_train, batch_size=50, epochs=50, verbose=1, validation_split=0.1)

# history_dict = history.history
# print(history_dict.keys())

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = np.arange(1, len(acc)+1)

plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, acc, label='Training accuracy')
plt.plot(epochs, val_acc, label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

y_test_pred = model.predict_classes(X_test_std, verbose=0)

from sklearn.metrics import roc_auc_score
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

print('a')