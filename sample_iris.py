# %%
from sklearn.model_selection import train_test_split
import pandas as pd
import mglearn
from sklearn.datasets import load_iris
# %%
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.DataFrame(data.target, columns=["Species"])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=77)
# %%

grr = pd.plotting.scatter_matrix(X, figsize=(15, 15), marker='x',
                                 hist_kwds={'bins': 20}, s=60)

# %%
# %%
