import pandas as pd
import smallestenclosingcircle
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("IRMA.csv")

data = df[df["safegraph_id"]=='c9c0b25594eaa645db562a4f410f80c32af583a11a6bcd4134d1eaece912c3b4']
X = data[['x','y']].values
db = DBSCAN(eps=2000, min_samples=2).fit(X) 
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
#temp = np.load('ids.npy')
#ids = dict((k,0) for k in temp)
#del temp
#j = 0
#for i in ids.keys():
#    if j%1000 == 0:
#        print j
#    j+=1
#    ids[i] = smallestenclosingcircle.make_circle(df[df.safegraph_id==i][['x','y']].values)[2]
