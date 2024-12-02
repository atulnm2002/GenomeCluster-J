import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

# Jaccard distance matrix
dist_matrix = np.array(
    [
        [0, 0.138, 0.114, 0.109, 0.079, 0.129, 0.119, 0.181, 0.079],
        [0.138, 0, 0.134, 0.12, 0.1, 0.161, 0.141, 0.202, 0.1],
        [0.114, 0.134, 0, 0.111, 0.082, 0.147, 0.13, 0.194, 0.082],
        [0.109, 0.12, 0.111, 0, 0.043, 0.135, 0.118, 0.175, 0.043],
        [0.079, 0.1, 0.082, 0.043, 0, 0.105, 0.091, 0.158, 0],
        [0.129, 0.161, 0.147, 0.135, 0.105, 0, 0.148, 0.199, 0.105],
        [0.119, 0.141, 0.13, 0.118, 0.091, 0.148, 0, 0.179, 0.091],
        [0.181, 0.202, 0.194, 0.175, 0.158, 0.199, 0.179, 0, 0.158],
        [0.079, 0.1, 0.082, 0.043, 0, 0.105, 0.091, 0.158, 0],
    ]
)

# Make the matrix symmetric
dist_matrix = (dist_matrix + dist_matrix.T) / 2

# Use MDS to convert the distance matrix into 2D points
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
points = mds.fit_transform(dist_matrix)

# Perform k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(points)

# Create the plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(points[:, 0], points[:, 1], c=labels, cmap="viridis")
plt.colorbar(scatter)
plt.title("K-means Clustering of COVID-19 Variants")
plt.xlabel("MDS Dimension 1")
plt.ylabel("MDS Dimension 2")

# Add labels for each point
variant_names = [
    "Beta",
    "Delta",
    "Epsilon",
    "First",
    "Fourth",
    "Gamma",
    "Lambda",
    "Omicron",
    "Second",
]
for i, (x, y) in enumerate(points):
    plt.annotate(variant_names[i], (x, y), xytext=(5, 5), textcoords="offset points")

plt.savefig("kmeans_third_test.png")
plt.close()

print("K-means plot saved")
