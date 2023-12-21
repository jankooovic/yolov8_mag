""" Implementacija PCA metode za odstranitev napačnih koordinat """
# https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
# https://visualstudiomagazine.com/articles/2021/10/20/anomaly-detection-pca.aspx

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# predelaj PCA metodo na točkah, ki jih imaš


# Its behavior is easiest to visualize by looking at a two-dimensional dataset. Consider the following 200 points
# tukaj naložim točke, ki so referenčne
rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
"""
plt.scatter(X[:, 0], X[:, 1])
plt.axis('equal');
"""
print("Points:", X)


# The fit learns some quantities from the data, most importantly the "components" and "explained variance":
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)


# PCA components
print(pca.components_)

# PCA variance
print(pca.explained_variance_)

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# plot data
"""
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
plt.axis('equal');
"""
from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape

pca = PCA().fit(digits.data)
"""
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
"""

def plot_digits(data):
    fig, axes = plt.subplots(4, 10, figsize=(10, 4),
                             subplot_kw={'xticks':[], 'yticks':[]},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8),
                  cmap='binary', interpolation='nearest',
                  clim=(0, 16))

"""
plot_digits(digits.data)
"""

# tukaj dodaj predikted točke 
np.random.seed(42)
noisy = np.random.normal(digits.data, 4)
"""
plot_digits(noisy)
"""

pca = PCA(0.50).fit(noisy)
print(pca.n_components_)

components = pca.transform(noisy)
filtered = pca.inverse_transform(components)
plot_digits(filtered)
plt.show()