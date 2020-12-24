# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston

# %% [markdown]
# # load_boston
# ## y = ax + b -> a,b の関数

# %% [markdown]
# ## 読み込み
# %%
dataset = load_boston()
samples, label, feature_names = dataset.data, dataset.target, dataset.feature_names

# %% [markdown]
# ## DataFrame にする
# %%
bostondf = pd.DataFrame(dataset.data)
bostondf.columns = dataset.feature_names
bostondf['target_price'] = dataset.target
bostondf
# %% [markdown]
# ## 散布図をプロット

# %%
bostondf.plot(x='RM', y='target_price', style='x')
plt.title('RM vs target price')
plt.ylabel('target price')
plt.xlabel('RM')
plt.show()
# %%


def predict(X, coeff, intercept):
    '''
    予測
    '''
    return X * coeff + intercept


def cost_function(X, y, coeff, intercept):
    '''
    誤差関数
    '''
    MSE = 0.0
    n = len(X)
    for i in range(n):
        MSE += (y[i] - (coeff * X[i] + intercept))**2
    return MSE / n


def update_weight(X, y, coeff, intercept, learning_rate):
    '''
    重み更新
    '''
    coeff_derivative = 0
    intercept_derivative = 0
    n = len(X)
    for i in range(n):
        coeff_derivative += -2 * X[i] * (y[i] - (coeff*X[i]+intercept))
        intercept_derivative += -2 * (y[i] - (coeff * X[i] + intercept))
    coeff -= (coeff_derivative / n) * learning_rate
    intercept -= (intercept_derivative / n) * learning_rate
    return coeff, intercept


def train(X, y, coeff, intercept, learning_rate, iteration):
    '''
    勾配降下法
    '''
    cost_hist = []
    for _ in range(iteration):
        coeff, intercept = update_weight(X, y, coeff, intercept, learning_rate)
        cost = cost_function(X, y, coeff, intercept)
        cost_hist.append(cost)
    return coeff, intercept, cost_hist


# %% [markdown]
# %%
X = bostondf.RM.values
y = bostondf.target_price.values

# %% [markdown]
# ## 学習してみる
# %%
coeff, intercept, cost_history = train(
    X=X, y=y, coeff=0.3, intercept=0.3, learning_rate=0.01, iteration=5000)
# %% [markdown]
# ## plot
# %%
y_pred = predict(X, coeff, intercept)
plt.plot(X, y, 'x')
plt.plot(X, y_pred)
plt.show()

# %%
# sklearn で線形回帰
# %%
# 分割
X_train, X_test, y_train, y_test = train_test_split(
    bostondf.drop('target_price', axis=1).values, bostondf.target_price.values, test_size=0.2, random_state=77)
# 線形モデル
model = LinearRegression()
model.fit(X_train, y_train)
# 予測結果
y_pred = model.predict(X_test)
# %% [markdown]
# ## plot
# %%
plt.scatter(y_test, y_pred)
plt.xlabel('price')
plt.ylabel('predicted price')
plt.title('y_test vs y_pred')
plt.axis('equal')
# %% [markdown]
