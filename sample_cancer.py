# %%
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# %%
data = load_breast_cancer()
X = data.data
y = data.target
# %%
X
# %%
y
# %%
# 平均値のデータを使用
X = X[:, :10]
# %%
X
# %%

# ロジスティック回帰
model = LogisticRegression()
# %%
model.fit(X, y)
# %%
y_pred = model.predict(X)
# %%
y_pred
# %%
y
# %%

accuracy_score(y, y_pred)
# %%
