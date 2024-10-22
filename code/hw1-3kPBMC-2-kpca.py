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
## 2. Kernel PCA
#################################

from sklearn.decomposition import KernelPCA

kpca = KernelPCA(n_components=10, kernel='rbf')

gamma_list =  [0.001, 0.01, 0.1, 1, 10] 

best_error = 10000
best_gamma = None

for gamma in gamma_list:
    print(f"Testing gamma = {gamma}")

    # Apply kpca with the current gamma value
    kpca = KernelPCA(n_components=10, kernel='rbf', gamma = gamma, fit_inverse_transform=True)
    
    # Transform the data using Spectral Embedding
    x_kpca = kpca.fit_transform(x)

    x_reconstructed = kpca.inverse_transform(x_kpca)
    
    # Step 4: Calculate the reconstruction error (Mean Squared Error)
    reconstruction_error = mean_squared_error(x, x_reconstructed)
    
    print(f"Reconstruction error for gamma ={gamma}: {reconstruction_error:.4f}")
    
    # Keep track of the best score and corresponding n_neighbors value
    if reconstruction_error < best_error:
        best_error = reconstruction_error
        best_gamma = gamma

# Step 5: Output the best n_neighbors value and corresponding accuracy
print(f"\nBest gamma: {best_gamma}")
print(f'Reconstruction Error (MSE): {best_error}')

kpca = KernelPCA(n_components=10, kernel='rbf', gamma = best_gamma)
    
# Transform the data using Spectral Embedding
x_kpca = kpca.fit_transform(x)

trust_kpca = trustworthiness(x, x_kpca, n_neighbors=5)
print(f'Trustworthiness 5 (KPCA): {trust_kpca}')

trust_kpca = trustworthiness(x, x_kpca, n_neighbors=20)
print(f'Trustworthiness 20 (KPCA): {trust_kpca}')

plot_2d(x_kpca, x["CST3"], "CST3", filename = 'hw1/plots_3kPBMC/2_KPCA_firsttwo_CST3.png', title="CST3 - KPCA - First Two Components - 3kPBMC")
plot_2d(x_kpca, x["NKG7"], "NKG7", filename = 'hw1/plots_3kPBMC/2_KPCA_firsttwo_NKG7.png', title="NKG7 - KPCA - First Two Components - 3kPBMC")
plot_2d(x_kpca, x["PPBP"], "PPBP", filename = 'hw1/plots_3kPBMC/2_KPCA_firsttwo_PPBP.png', title="PPBP - KPCA - First Two Components - 3kPBMC")
