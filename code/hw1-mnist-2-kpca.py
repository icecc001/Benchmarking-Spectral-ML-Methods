## BIOS774 HW1 Benchmarking Spectral Methods MNIST KPCA
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

#################################
## 2. Kernel PCA
#################################

from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

kpca = KernelPCA(n_components=30, kernel='rbf')
svm_model = SVC(kernel='linear', random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kpca', kpca),
    ('svm', svm_model)
])

# Parameters to tune
param_grid = {
    'kpca__gamma': [0.001, 0.01, 0.1, 1, 10]  # Tuning gamma for RBF
}

# Using GridSearchCV to find the best parameters
grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy')
grid_search.fit(sampled_x, sampled_y_encoded)
print("Best Parameters:", grid_search.best_params_)


## I feel like for sepctual methods to be comparable, we should use the same number of components
scaler = StandardScaler()
sampled_x_scaled = scaler.fit_transform(sampled_x)  # Standardize the x values

kpca = KernelPCA(n_components=30, gamma = grid_search.best_params_['kpca__gamma'], kernel='rbf')

# Apply Kernel PCA with the RBF kernel
x_kpca = kpca.fit_transform(sampled_x_scaled)

svm_model = SVC(kernel='linear', random_state=42)

# Perform 10-fold cross-validation
# cv=10 specifies 10-fold cross-validation
cv_scores = cross_val_score(svm_model, x_kpca, sampled_y_encoded, cv=10, scoring='accuracy')

# Calculate and print the total accuracy
total_accuracy = cv_scores.mean()
print(f"SVM Prediction Accuracy after KPCA with 10-fold cross-validation: {total_accuracy:.4f}")

## plots
plot_2d(x_kpca, sampled_y_encoded, filename = f'hw1/plots_{nsample}/kpca_mnist_firsttwo.png', title="KPCA - First Two Components - MNIST Data")

