
# %%
from sklearn.datasets import load_wine
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

# %%
data = load_wine()
df_X = pd.DataFrame(data.data, columns=data.feature_names)
# %%
df_X.head()

# %%
df_y = pd.DataFrame(data.target, columns=["kind(target)"])

# %%
df_y
# %%
data.target
# %%
df = pd.concat([df_X, df_y], axis=1)

# %%
df
# %%
plt.hist(df.loc[:, "alcohol"])
# %%
plt.boxplot(df.loc[:, "alcohol"])

# %%
type(df)
# %%
type(df[:])
# %%
df.corr()
# %%
type(df.loc[:, "alcohol"])
# %%

df.describe()
# %%

_ = scatter_matrix(df, figsize=(15, 15))
# %%
_ = scatter_matrix(df.iloc[:, [0, 9, -1]])
# %%
