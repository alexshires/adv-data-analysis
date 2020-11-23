"""
Example for KDE
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

np.random.seed(0)
N = 200
X = np.concatenate((np.random.normal(0, 1, int(0.3 * N)),
                    np.random.normal(5, 1, int(0.7 * N))))[:, np.newaxis]
X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
bins = np.linspace(-5, 10, 10)


plt.hist(X[:, 0], density=True, bins=20)
plt.savefig("histogram.png")

plt.clf()
# Gaussian kernel density
kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(X)
log_dens = kde.score_samples(X_plot)
color = 'navy'
lw = 2
plt.hist(X[:, 0], density=True, bins=20)
plt.plot(X_plot[:, 0], np.exp(log_dens), color=color, lw=lw, linestyle='-', label="kernel = 'Gaussian'")

plt.savefig("histogram_kde.png")


