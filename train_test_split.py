# %%
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import numpy as np

a = np.arange(10)
print(a)
# %%
# [0 1 2 3 4 5 6 7 8 9]
gr = np.array([3, 3, 3, 1], dtype=int)
# %%
gss = GroupShuffleSplit(n_splits=2, train_size=.7, random_state=42)
# %%
print(gss.split(a, groups=gr))
# %%
# [array([3, 9, 6, 1, 5, 0, 7]), array([2, 8, 4])]

# %%
x = next(gss.split(a, groups=gr))

# %%
