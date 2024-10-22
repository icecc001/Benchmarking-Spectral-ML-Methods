## BIOS774 HW1 Benchmarking Spectral Methods (Data Two)
# Author: Xinyu Zhang
# Date: Sept. 30th

# Import Packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import trustworthiness
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

#####################################
## 6. Diffusion Map
#####################################

from pydiffmap import diffusion_map as dm

alpha_list = [0, 0.5, 1]
epsilon_list = [0.1, 1, 10]

best_trust = 0
best_alpha = None

for alpha in alpha_list:

    print(f"Testing alpha = {alpha}")

    diffusion_map = dm.DiffusionMap.from_sklearn(n_evecs=10, epsilon='bgh', alpha=alpha)

    # Transform the data using Diffusion Map
    try:
        # Attempt to fit and transform the data using diffusion map
        x_diffusion = diffusion_map.fit_transform(x)
    except Exception as e:
        # If an error occurs (such as convergence failure), print the error and skip
        print(f"Error: {e}. Skipping this iteration.")
        x_diffusion = None 
        continue
    if np.isnan(x_diffusion.max()):
        continue
    
    trust = trustworthiness(x, x_diffusion, n_neighbors=5)
    
    print(f'Trustworthiness 5: {trust}')
    
    # Keep track of the best trust and corresponding n_neighbors value
    if trust > best_trust:
        best_trust = trust
        best_alpha = alpha

# Output the best n_neighbors value and corresponding RE
print(f"\nBest alpha: {best_alpha}")
print(f'Trustworthiness 5: {best_trust}')

diffusion_map = dm.DiffusionMap.from_sklearn(n_evecs=10, epsilon='bgh', alpha=best_alpha)

# Transform the data using Spectral Embedding
x_diffusion = diffusion_map.fit_transform(x)

trust_diffusion5 = trustworthiness(x, x_diffusion, n_neighbors=5)
print(f'Trustworthiness 5 (LE): {trust_diffusion5}')

trust_diffusion20 = trustworthiness(x, x_diffusion, n_neighbors=20)
print(f'Trustworthiness 20 (LE): {trust_diffusion20}')

plot_2d(x_diffusion, x["CST3"], "CST3", filename = 'hw1/plots_3kPBMC/6_diffusionmap_firsttwo_CST3.png', title="CST3 - Diffusion Map - First Two Components - 3kPBMC")
plot_2d(x_diffusion, x["NKG7"], "NKG7", filename = 'hw1/plots_3kPBMC/6_diffusionmap_firsttwo_NKG7.png', title="NKG7 - Diffusion Map - First Two Components - 3kPBMC")
plot_2d(x_diffusion, x["PPBP"], "PPBP", filename = 'hw1/plots_3kPBMC/6_diffusionmap_firsttwo_PPBP.png', title="PPBP - Diffusion Map - First Two Components - 3kPBMC")
