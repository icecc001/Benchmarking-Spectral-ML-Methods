## BIOS774 HW1 Benchmarking Spectral Methods (Data Two)
# Author: Xinyu Zhang
# Date: Sept. 30th

# Import Packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import trustworthiness
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import pairwise_distances


# Read CSV file into a DataFrame
x = pd.read_csv('hw1/data/3kPBMC.csv')

# Plot the first two dimensions of the PCA-reduced data and color them by y
def plot_2d(x_pca, y, yname, filename, title):
    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label= yname)
    
    plt.title(title)
    plt.xlabel("First Component")
    plt.ylabel("Second Component")
    plt.grid(True)
    plt.savefig(filename, dpi = 300)

#################################
## 3. Isomap
#################################

from sklearn.manifold import Isomap

n_neighbors_list = [5, 10, 15, 20]

best_error = 10000
best_n_neighbors = None

for n_neighbors in n_neighbors_list:

    print(f"Testing n_neighbors = {n_neighbors}")

    # Apply isomap with the current n_neighbors value
    isomap = Isomap(n_components = 10, n_neighbors = n_neighbors)
    
    # Transform the data using isomap
    x_isomap = isomap.fit_transform(x)
    D_original = isomap.dist_matrix_
    
    # Calculate the pairwise distances in the reduced space
    D_reduced = pairwise_distances(x_isomap)
    
    # Calculate the reconstruction error (e.g., Mean Squared Error) ## not the same reconstruction error
    reconstruction_error = np.mean((D_original - D_reduced) ** 2)

    print(f"Reconstruction error for n_neighbors ={n_neighbors}: {reconstruction_error:.4f}")
    
    # Keep track of the best error and corresponding n_neighbors value
    if reconstruction_error < best_error:
        best_error = reconstruction_error
        best_n_neighbors = n_neighbors

# Step 5: Output the best n_neighbors value and corresponding RE
print(f"\nBest n_neighbors: {best_n_neighbors}")
print(f'Reconstruction Error (MSE): {best_error}')

isomap = Isomap(n_components = 10, n_neighbors = best_n_neighbors)
    
# Transform the data using Spectral Embedding
x_isomap = isomap.fit_transform(x)

trust_isomap5 = trustworthiness(x, x_isomap, n_neighbors=5)
print(f'Trustworthiness 5 (Isomap): {trust_isomap5}')

trust_isomap20 = trustworthiness(x, x_isomap, n_neighbors=20)
print(f'Trustworthiness 20 (Isomap): {trust_isomap20}')

plot_2d(x_isomap, x["CST3"], "CST3", filename = 'hw1/plots_3kPBMC/3_Isomap_firsttwo_CST3.png', title="CST3 - Isomap - First Two Components - 3kPBMC")
plot_2d(x_isomap, x["NKG7"], "NKG7", filename = 'hw1/plots_3kPBMC/3_Isomap_firsttwo_NKG7.png', title="NKG7 - Isomap - First Two Components - 3kPBMC")
plot_2d(x_isomap, x["PPBP"], "PPBP", filename = 'hw1/plots_3kPBMC/3_Isomap_firsttwo_PPBP.png', title="PPBP - Isomap - First Two Components - 3kPBMC")
