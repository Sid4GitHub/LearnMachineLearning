import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import threading


from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=40, centers=4,cluster_std=0.40, random_state=0)

X = X[:, ::-1]

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap='viridis');


from sklearn.cluster import KMeans
kmeans_model = KMeans(4, random_state=0)
labels = kmeans_model.fit(X).predict(X)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels, s=20, cmap='viridis');

#x = threading.Thread(target=lambda :plt.show())
#x.start()


from sklearn.mixture import GaussianMixture
gmm_model = GaussianMixture (n_components=4).fit(X)
labels = gmm_model.predict(X)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels, s=20, cmap='viridis');


plt.show()