# %% ライブラリの読み込み
import pandas as pd
import numpy as np
import warnings
import tensorflow as tf
# %%
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
boston = load_boston()

X = pd.DataFrame(data=boston.data, columns=boston.feature_names)
scaler = StandardScaler()
X[:] = scaler.fit_transform(X)
y = pd.DataFrame(data=boston.target, columns=['price'])

# 訓練データと評価データに分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=1)

# 訓練データと評価データに分割
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, train_size=0.8, random_state=1)

# Yeo-Johonson
pt = PowerTransformer(method='yeo-johnson')
X_train[:] = pt.fit_transform(X_train)
X_val[:] = pt.transform(X_val)

# %% Optimizerとepochの設定
optimizer = keras.optimizers.Adam(lr=0.02)
epochs = 100

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation=tf.nn.relu, input_shape=(11, )),
    tf.keras.layers.Dense(32, activation=tf.nn.relu),
    tf.keras.layers.Dense(32, activation=tf.nn.relu),
    tf.keras.layers.Dense(32, activation=tf.nn.relu),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer=optimizer,
              loss='mse',
              metrics=['mse'])

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, verbose=1)

history = model.fit(X_train, y_train.values, batch_size=256,
                    epochs=epochs, validation_data=(X_val, y_val.values))

# 予測値の計算
y_pred = model.predict(X_test).flatten()
# モデルの評価
model.evaluate(X_test, y_test.values)

# R2スコアとMAEスコアの表示
print("---------------------------")
print(str(optimizer).split('.')[-1].split(' ')[0] + ' score')
print("---------------------------")
print("R2 on test data  : %f" % r2_score(y_pred, y_test))
print("MAE on test data : %f" % mean_absolute_error(y_pred, y_test))

history_dict = history.history
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

# 学習経過をプロット
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# %%
