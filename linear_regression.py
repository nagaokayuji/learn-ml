# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston

# %% [markdown]
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
# %% [markdown]

# %%
# %% [markdown]

# %%
