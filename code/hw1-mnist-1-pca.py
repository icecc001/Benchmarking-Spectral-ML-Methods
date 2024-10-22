## BIOS774 HW1 Benchmarking Spectral Methods MNIST PCA
# Author: Xinyu Zhang
# Date: Sept. 28th

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

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
    #plt.show()

#################################
## 1. PCA
#################################

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# See the Variance Explained by PCA results with different number of components
# Perform PCA without specifying the number of components
pca = PCA()
scaler = StandardScaler()
sampled_x_scaled = scaler.fit_transform(sampled_x)  # Standardize the x values
pca.fit(sampled_x_scaled)

# Get cumulative explained variance
explained_variance = pca.explained_variance_ratio_.cumsum()

# Plot cumulative explained variance, cutoff at the point where 50% of variance is explained.
n_components_50 = np.argmax(explained_variance >= 0.5) + 1

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
# Add a horizontal line at 50% explained variance
plt.axhline(y=0.5, color='r', linestyle='-', label='50% Variance Explained')
plt.axvline(x=n_components_50, color='r', linestyle='--', label=f'{n_components_50} Components')

plt.title('Cumulative Explained Variance by Number of Principal Components: MNIST Data')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.legend()
plt.savefig(f"hw1/plots_{nsample}/pca_mnist_totalvar.png", dpi = 300)

# Perform PCA for dimensionality reduction
pca = PCA(n_components=30)
sampled_x_pca = pca.fit_transform(sampled_x_scaled)

svm_model = SVC(kernel='linear', random_state=42)

# Perform 10-fold cross-validation
# cv=10 specifies 10-fold cross-validation
cv_scores = cross_val_score(svm_model, sampled_x_pca, sampled_y_encoded, cv=10, scoring='accuracy')

# Calculate and print the total accuracy
total_accuracy = cv_scores.mean()
print(f"SVM Prediction Accuracy after PCA with 10-fold cross-validation: {total_accuracy:.4f}")

# Call the function to plot the first two component
plot_2d(sampled_x_pca, sampled_y_encoded, filename = f'hw1/plots_{nsample}/pca_mnist_firsttwo.png', title="PCA - First Two Components - MNIST Data")