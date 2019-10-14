 import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")

dataset = pd.read_csv('CC.csv')
x = dataset.iloc[:,1:]
# y = dataset.iloc[:,-1]

z = x.apply(lambda x: x.fillna(x.mean()),axis=0)

##building the model
from sklearn.cluster import KMeans
nclusters = 3 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(z)


wcss= []  ##Within Cluster Sum of Squares
##elbow method to know the number of clusters
for i in range(1,11):
    kmeans= KMeans(n_clusters=i,
max_iter=300,random_state=0)
    kmeans.fit(z)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

#Silhouette score
# predict the cluster for each data point
y_cluster_kmeans = km.predict(z)
from sklearn import metrics
score = metrics.silhouette_score(z, y_cluster_kmeans)
print("The Silhouette score is",score)

from sklearn import preprocessing
scaler =preprocessing.StandardScaler()
scaler.fit(z)
X_scaled_array=scaler.transform(z)
X_scaled=pd.DataFrame(X_scaled_array, columns =z.columns)

print("Feature Scaling",X_scaled)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(z)#
#Apply transform to both the training set and the test set.
x_scaler= scaler.transform(z)

from sklearn.decomposition import PCA# Make an instance of the Model
pca= PCA(2)
X_pca= pca.fit_transform(x_scaler)
print("Applying PCA on the dataset:",X_pca)
