import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Your distance matrix
dist_matrix = np.array([
    [0,     0.138,  0.114,  0.129,  0.119,  0.181,  0.181,  0.795],
    [0.138, 0,      0.134,  0.161,  0.141,  0.202,  0.202,  0.795],
    [0.114, 0.134,  0,      0.147,  0.13,   0.194,  0.194,  0.797],
    [0.129, 0.161,  0.147,  0,      0.148,  0.199,  0.199,  0.793],
    [0.119, 0.141,  0.13,   0.148,  0,      0.179,  0.179,  0.797],
    [0.181, 0.202,  0.194,  0.199,  0.179,  0,      0,      0.794],
    [0.181, 0.202,  0.194,  0.199,  0.179,  0,      0,      0.794],
    [0.795, 0.795,  0.797,  0.793,  0.797,  0.794,  0.794,  0]
])

# Convert the distance matrix to a condensed distance vector
condensed_dist = []
for i in range(len(dist_matrix)):
    for j in range(i+1, len(dist_matrix)):
        condensed_dist.append(dist_matrix[i][j])

# Perform hierarchical clustering
linkage_matrix = linkage(condensed_dist, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.savefig('dendrogram.png')
print("Dendrogram saved as 'dendrogram.png'")

# Print linkage matrix
print("Linkage matrix:")
print(linkage_matrix)

