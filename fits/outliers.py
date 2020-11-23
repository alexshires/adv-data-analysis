"""
Outlier example
"""



import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

np.random.seed(0)
N = 2000
FRAC = 0.98
X = np.concatenate((np.random.normal(0, 1, int(FRAC * N)),
                    np.random.normal(5, 1, int((1.- FRAC) * N))))[:, np.newaxis]
X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]

plt.hist(X[:, 0], density=True, bins=20, label='data')
# use scipy .. stas
res1 = norm.fit(X)
print(norm, res1)
rv = norm()

# print
plt.plot(X_plot[:, 0], rv.pdf(X_plot), label='fit')
plt.savefig("outlier_histogram_fit.png")


