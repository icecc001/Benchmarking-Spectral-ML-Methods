## BIOS774 HW1 Benchmarking Spectral Methods MNIST diffusion map
# Author: Xinyu Zhang
# Date: Sept. 28th

import pandas as pd
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

nsample = 5000

# Read CSV file into a DataFrame
sampled_data = pd.read_csv(f'hw1/data/sample_data_{nsample}_standardized.csv')

# If you want to split back into x and y after sampling
sampled_x = sampled_data.iloc[:, :-1]  
sampled_y = sampled_data.iloc[:, -1]

# Convert y to categorical labels using LabelEncoder
label_encoder = LabelEncoder()
sampled_y_encoded = label_encoder.fit_transform(sampled_y)


# Plot the first two dimensions of the PCA-reduced data and color them by y
def plot_2d(x_pca, y, filename, title):
    plt.figure(figsize=(8, 6))

    # Get unique categories
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)

    # Use viridis color map but with discrete steps
    viridis = plt.cm.get_cmap('tab10', num_classes)
    cmap = ListedColormap(viridis(np.arange(num_classes)))

    scatter = plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y, cmap=cmap, alpha=0.7)

    # Set color boundaries
    norm = BoundaryNorm(np.arange(num_classes + 1) - 0.5, num_classes)

    # Add color bar and set ticks to categorical labels
    cbar = plt.colorbar(scatter, ticks=np.arange(num_classes), norm=norm, label="Labels")
    cbar.set_ticks(np.arange(num_classes))
    cbar.set_ticklabels(unique_classes)
    
    plt.title(title)
    plt.xlabel("First Component")
    plt.ylabel("Second Component")
    plt.grid(True)
    plt.savefig(filename, dpi = 300)

# svm model
svm_model = SVC(kernel='linear', random_state=42)


#################################
## 6. Diffusion Map
#################################

from pydiffmap import diffusion_map as dm
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler


# Apply Diffusion Map for dimensionality reduction
diffusion_map = dm.DiffusionMap.from_sklearn(n_evecs=30)

#epsilon_list = [0.1, 1, 10]
alpha_list = [0, 0.5, 1] 
neighbor_params = {'n_jobs': -1, 'algorithm': 'ball_tree'}
best_score = 0
best_params = None

# Create 10-fold cross-validation splits
kf = KFold(n_splits=10)
svm_model = SVC(kernel='linear', random_state=42)


scaler = MinMaxScaler()
sampled_x_scaled = scaler.fit_transform(sampled_x)


# Loop through the parameter grid
for alpha in alpha_list:  # Now tuning alpha as well
    print(f"Testing alpha = {alpha}")
        
    # Apply Diffusion Map with current n_eigenpairs, epsilon, and alpha
    diffusion_map = dm.DiffusionMap.from_sklearn(n_evecs=30, epsilon='bgh', alpha=alpha, neighbor_params=neighbor_params)
            
    # Transform the data using Diffusion Map
    try:
        # Attempt to fit and transform the data using diffusion map
        X_diffusion = diffusion_map.fit_transform(sampled_x_scaled)
    except Exception as e:
        # If an error occurs (such as convergence failure), print the error and skip
        print(f"Error: {e}. Skipping this iteration.")
        X_diffusion = None 
        continue
    if np.isnan(X_diffusion.max()):
        continue
        
    # Perform 10-fold cross-validation with the transformed data
    scores = cross_val_score(svm_model, X_diffusion, sampled_y_encoded, cv=kf, scoring='accuracy')
            
    # Compute the mean score across all folds
    mean_score = np.mean(scores)
        
    print(f"Mean Accuracy for alpha={alpha}: {mean_score:.4f}")
            
    # Keep track of the best score and corresponding parameters
    if mean_score > best_score:
        best_score = mean_score
        best_params = {'alpha': alpha}

print(f"\nBest Parameters: {best_params}")
print(f"Best Cross-Validation Accuracy: {best_score:.4f}")

diffusion_map = dm.DiffusionMap.from_sklearn(n_evecs=30, epsilon = 'bgh', alpha = best_params['alpha'])

x_diffmap = diffusion_map.fit_transform(sampled_x_scaled)


# Perform 10-fold cross-validation
# cv=10 specifies 10-fold cross-validation
cv_scores = cross_val_score(svm_model, x_diffmap, sampled_y_encoded, cv=10, scoring='accuracy')

# Calculate and print the total accuracy
total_accuracy = cv_scores.mean()
print(f"SVM Prediction Accuracy after diffusion map with 10-fold cross-validation: {total_accuracy:.4f}")

## plots
plot_2d(x_diffmap, sampled_y_encoded, filename = f'hw1/plots_{nsample}/Diffusion_mnist_firsttwo.png', title="Diffusion Map - First Two Components - MNIST Data")
