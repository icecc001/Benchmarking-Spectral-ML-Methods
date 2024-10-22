## BIOS774 HW1 Benchmarking Spectral Methods MNIST PCA
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
sampled_data = pd.read_csv(f'hw1/data/sample_data_{nsample}.csv')

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
## 5. Laplacian Eigenmap
#################################

from sklearn.manifold import SpectralEmbedding

# Apply Laplacian Eigenmap (spectral embedding) for dimensionality reduction
laplacian = SpectralEmbedding(n_components=30, affinity='nearest_neighbors')

n_neighbors_list = [5, 10, 15, 20]

best_score = 0
best_n_neighbors = None
scaler = StandardScaler()
sampled_x_scaled = scaler.fit_transform(sampled_x)

for n_neighbors in n_neighbors_list:
    print(f"Testing n_neighbors = {n_neighbors}")

    # Apply Laplacian Eigenmap (Spectral Embedding) with the current n_neighbors value
    laplacian = SpectralEmbedding(n_components=30, n_neighbors=n_neighbors, affinity='nearest_neighbors')
    
    # Transform the data using Spectral Embedding
    x_laplacian = laplacian.fit_transform(sampled_x_scaled)
    
    # Perform 10-fold cross-validation on the SVM using the transformed data
    scores = cross_val_score(svm_model, x_laplacian, sampled_y_encoded, cv=10, scoring='accuracy')
    
    # Compute the mean score across all folds
    mean_score = np.mean(scores)
    
    print(f"Mean Accuracy for n_neighbors={n_neighbors}: {mean_score:.4f}")
    
    # Keep track of the best score and corresponding n_neighbors value
    if mean_score > best_score:
        best_score = mean_score
        best_n_neighbors = n_neighbors

# Step 5: Output the best n_neighbors value and corresponding accuracy
print(f"\nBest n_neighbors: {best_n_neighbors}")
#print(f"Best Cross-Validation Accuracy: {best_score:.4f}")

laplacian = SpectralEmbedding(n_components = 30, n_neighbors = best_n_neighbors, affinity='nearest_neighbors')

# Apply Isomap
x_laplacian = laplacian.fit_transform(sampled_x_scaled)

svm_model = SVC(kernel='linear', random_state=42)

# Perform 10-fold cross-validation
# cv=10 specifies 10-fold cross-validation
cv_scores = cross_val_score(svm_model, x_laplacian, sampled_y_encoded, cv=10, scoring='accuracy')

# Calculate and print the total accuracy
total_accuracy = cv_scores.mean()
print(f"SVM Prediction Accuracy after Laplacian Eigenmap with 10-fold cross-validation: {total_accuracy:.4f}")

## plots
plot_2d(x_laplacian, sampled_y_encoded, filename = f'hw1/plots_{nsample}/Laplacian_std_mnist_firsttwo.png', title="Laplacian Eigenmap - First Two Components - MNIST")
