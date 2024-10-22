## BIOS774 HW1 Benchmarking Spectral Methods (Data Two)
# Author: Xinyu Zhang
# Date: Sept. 30th

# Import Packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import trustworthiness

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

#####################################
## 7. Hessian LLE
#####################################

from sklearn.manifold import LocallyLinearEmbedding

n_neighbors_list = [70, 100, 150]

best_trust = 0
best_n_neighbors = None

for n_neighbors in n_neighbors_list:

    print(f"Testing n_neighbors = {n_neighbors}")

    # Apply lle with the current n_neighbors value
    hlle = LocallyLinearEmbedding(n_components = 10, method = "hessian", n_neighbors = n_neighbors)

    x_hlle = hlle.fit_transform(x)
    
    trust = trustworthiness(x, x_hlle, n_neighbors=5)
    
    print(f'Trustworthiness 5: {trust}')
    
    # Keep track of the best trust and corresponding n_neighbors value
    if trust > best_trust:
        best_trust = trust
        best_n_neighbors = n_neighbors

# Output the best n_neighbors value and corresponding RE
print(f"\nBest n_neighbors: {best_n_neighbors}")
print(f'Trustworthiness 5: {best_trust}')

hlle = LocallyLinearEmbedding(n_components = 10, n_neighbors = best_n_neighbors, method = "hessian")

# Transform the data using Spectral Embedding
x_hlle = hlle.fit_transform(x)

trust_hlle5 = trustworthiness(x, x_hlle, n_neighbors=5)
print(f'Trustworthiness 5 (HLLE): {trust_hlle5}')

trust_hlle20 = trustworthiness(x, x_hlle, n_neighbors=20)
print(f'Trustworthiness 20 (HLLE): {trust_hlle20}')

plot_2d(x_hlle, x["CST3"], "CST3", filename = 'hw1/plots_3kPBMC/7_HLLE_firsttwo_CST3.png', title="CST3 - Hessian LLE - First Two Components - 3kPBMC")
plot_2d(x_hlle, x["NKG7"], "NKG7", filename = 'hw1/plots_3kPBMC/7_HLLE_firsttwo_NKG7.png', title="NKG7 - Hessian LLE - First Two Components - 3kPBMC")
plot_2d(x_hlle, x["PPBP"], "PPBP", filename = 'hw1/plots_3kPBMC/7_HLLE_firsttwo_PPBP.png', title="PPBP - Hessian LLE - First Two Components - 3kPBMC")
