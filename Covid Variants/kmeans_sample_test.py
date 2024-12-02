import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import MDS


if not os.path.exists('Pictures'):
    os.makedirs('Pictures')
    
# Jaccard distance matrix
dist_matrix = np.array([
    [0,     0.138,  0.114,  0.129,  0.119,  0.181,  0.109],
    [0.138, 0,      0.134,  0.161,  0.141,  0.202,  0.12 ],
    [0.114, 0.134,  0,      0.147,  0.13,   0.194,  0.111],
    [0.129, 0.161,  0.147,  0,      0.148,  0.199,  0.135],
    [0.119, 0.141,  0.13,   0.148,  0,      0.179,  0.118],
    [0.181, 0.202,  0.194,  0.199,  0.179,  0,      0.175],
    [0.109, 0.12,   0.111,  0.135,  0.118,  0.175,  0    ]
])

# Make the matrix symmetric
dist_matrix = (dist_matrix + dist_matrix.T) / 2

# Use MDS to convert the distance matrix into 2D points
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
points = mds.fit_transform(dist_matrix)

# Perform k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(points)

# Create the plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis')
plt.colorbar(scatter)
plt.title('K-means Clustering of COVID-19 Variants')
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')

# Add labels for each point
variant_names = ['Alpha', 'Delta', 'Epsilon', 'Gamma', 'Lambda', 'Omicron', 'First']
for i, (x, y) in enumerate(points):
    plt.annotate(variant_names[i], (x, y), xytext=(5, 5), textcoords='offset points')

plt.savefig('Pictures/kmeans_Sample_test_first.png')
plt.close()

print("K-means plot saved as 'kmeans_plot.png'")