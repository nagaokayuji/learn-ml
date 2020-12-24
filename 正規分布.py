# %%
import matplotlib.pyplot as plt
import numpy as np

# %%

# f(x) = ( 1 / ( sqrt(2*pi))* sigma) * exp(- (x - mu)^2 / (2 sigma^2))

x = np.linspace(-100, 100, 300)
# %%
sigma = 20
mu = 0
# %%
# %%
y = 1. / ((2*np.pi)**0.5 * sigma) * np.exp(- ((x - mu)**2) / (2 * sigma**2))

plt.plot(x, y)
# %%
# %% [markdown]
# # 乱数を生成
x = np.random.normal(50, 10, 1000)
# %% [markdown]
# # ヒストグラムを出力
# %%
plt.hist(x, bins=20)

# %%
