## BIOS774 HW1 Benchmarking Spectral Methods (Data Two)
# Author: Xinyu Zhang
# Date: Sept. 30th

# Import Packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import trustworthiness
from sklearn.metrics import mean_squared_error
import numpy as np

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
## 1. PCA
#################################

from sklearn.decomposition import PCA

# See the Variance Explained by PCA results with different number of components
# Perform PCA without specifying the number of components
pca = PCA()
pca.fit(x)

# Get cumulative explained variance
explained_variance = pca.explained_variance_ratio_.cumsum()

# Perform PCA for dimensionality reduction
pca = PCA(n_components=10)
x_pca = pca.fit_transform(x)

x_reconstructed = pca.inverse_transform(x_pca)

# Step 4: Calculate the reconstruction error (Mean Squared Error)
reconstruction_error = mean_squared_error(x, x_reconstructed)

print(f'Reconstruction Error (MSE): {reconstruction_error}')

trust_pca5 = trustworthiness(x, x_pca, n_neighbors=5)
print(f'Trustworthiness 5 (PCA): {trust_pca5}')

trust_pca20 = trustworthiness(x, x_pca, n_neighbors=20)
print(f'Trustworthiness 20 (PCA): {trust_pca20}')

plot_2d(x_pca, x["CST3"], "CST3", filename = 'hw1/plots_3kPBMC/1_PCA_firsttwo_CST3.png', title="CST3 - PCA - First Two Components - 3kPBMC")
plot_2d(x_pca, x["NKG7"], "NKG7", filename = 'hw1/plots_3kPBMC/1_PCA_firsttwo_NKG7.png', title="NKG7 - PCA - First Two Components - 3kPBMC")
plot_2d(x_pca, x["PPBP"], "PPBP", filename = 'hw1/plots_3kPBMC/1_PCA_firsttwo_PPBP.png', title="PPBP - PCA - First Two Components - 3kPBMC")