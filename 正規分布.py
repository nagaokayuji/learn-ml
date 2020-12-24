# %%
import matplotlib.pyplot as plt
import numpy as np

# %%

# f(x) = ( 1 / ( sqrt(2*pi))* sigma) * exp(- (x - mu)^2 / (2 sigma^2))

x = np.linspace(-100, 100, 300)
# %%
sigma = 5
mu = 5
# %%
x
# %%
f = 1. / ((2*np.pi)**0.5 * sigma) * np.exp(- ((x - mu)**2) / (2 * sigma**2))

# %%

plt.plot(x, f)
# %%
# %%
