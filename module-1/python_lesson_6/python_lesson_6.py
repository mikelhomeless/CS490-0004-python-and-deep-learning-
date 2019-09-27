import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

dataset = pd.read_csv('CC.csv')
# remove the customer id from the dataset
x = dataset.iloc[:, 1:]
x.fillna(x.mean(), inplace=True)

# Question 1
wcss = []
for i in range(1, 15):
    km = KMeans(n_clusters=i)
    km.fit(x)
    wcss.append(km.inertia_)

# 7 was the most optimal
# Question 2 silhouette score
km = KMeans(n_clusters=7)
km.fit(x)
y_cluster = km.predict(x)
score = metrics.silhouette_score(x, y_cluster)
print("Silhouette score for 7 clusters is:", score)

# Question 3 Feature Scaling
scaler = StandardScaler()
scaler.fit(x)
x_scaled_array = scaler.transform(x)
x_scaled = pd.DataFrame(x_scaled_array, columns = x.columns)
km = KMeans(n_clusters=7)
km.fit(x_scaled)
y_cluster = km.predict(x_scaled)
score = metrics.silhouette_score(x_scaled, y_cluster)
print("Silhouette score for scaled data at 7 clusters is:", score)

# Question 4 PCA
pca = PCA(2)
x_pca = pca.fit_transform(x_scaled_array)
df_pca = pd.DataFrame(data=x_pca)
print("PCA data-set is:", df_pca)

# Bonus
km = KMeans(n_clusters=7)
km.fit(df_pca)
y_cluster = km.predict(df_pca)
score = metrics.silhouette_score(x, y_cluster)
print("Silhouette score for PCA data-set is:", score)
# After running kmeans on the pca data-set, it appears that the score is much lower than the previous scores we had encountered meaning this
# data does not benefit from our PCA

plt.plot(range(1, 15), wcss, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Wcss')
plt.show()
