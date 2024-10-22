## BIOS774 HW1 Benchmarking Spectral Methods
# Author: Xinyu Zhang
# Date: Sept. 28th

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

nsample = 5000

# Read CSV file into a DataFrame
sampled_data = pd.read_csv(f'hw1/data/sample_data_{nsample}.csv')

# If you want to split back into x and y after sampling
sampled_x = sampled_data.iloc[:, :-1]  
sampled_y = sampled_data.iloc[:, -1]


# Convert y to categorical labels using LabelEncoder
label_encoder = LabelEncoder()
sampled_y_encoded = label_encoder.fit_transform(sampled_y)


#################################
## 1. PCA
#################################

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# See the Variance Explained by PCA results with different number of components
# Perform PCA without specifying the number of components
pca = PCA()
pca.fit(sampled_x)

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

# Plot explained variance ## Scree Plot
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.title('Explained Variance by Principal Components (Scree Plot): MNIST Data')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance')
plt.grid(True)
plt.savefig(f"hw1/plots_{nsample}/pca_mnist_screevar.png", dpi = 300)

# Perform PCA for dimensionality reduction
pca = PCA(n_components=30)
sampled_x_pca = pca.fit_transform(sampled_x)

svm_model = SVC(kernel='linear', random_state=42)

# Perform 10-fold cross-validation
# cv=10 specifies 10-fold cross-validation
cv_scores = cross_val_score(svm_model, sampled_x_pca, sampled_y_encoded, cv=10, scoring='accuracy')

# Calculate and print the total accuracy
total_accuracy = cv_scores.mean()
print(f"SVM Prediction Accuracy after PCA with 10-fold cross-validation: {total_accuracy:.4f}")

# Plot the first two dimensions of the PCA-reduced data and color them by y
def plot_2d(x_pca, y, filename, title):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label="Labels")
    plt.title(title)
    plt.xlabel("First Component")
    plt.ylabel("Second Component")
    plt.grid(True)
    plt.savefig(filename, dpi = 300)
    #plt.show()

# Call the function after running PCA to get 
plot_2d(sampled_x_pca, sampled_y_encoded, filename = f'hw1/plots_{nsample}/pca_mnist_firsttwo.png', title="PCA - First Two Components - MNIST Data")


#################################
## 2. Kernel PCA
#################################

from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

kpca = KernelPCA(n_components=30, kernel='rbf')
svm_model = SVC(kernel='linear', random_state=42)

pipeline = Pipeline([
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
kpca = KernelPCA(n_components=30, gamma = grid_search.best_params_['kpca__gamma'], kernel='rbf')

# Apply Kernel PCA with the RBF kernel
x_kpca = kpca.fit_transform(sampled_x)

svm_model = SVC(kernel='linear', random_state=42)

# Perform 10-fold cross-validation
# cv=10 specifies 10-fold cross-validation
cv_scores = cross_val_score(svm_model, x_kpca, sampled_y_encoded, cv=10, scoring='accuracy')

# Calculate and print the total accuracy
total_accuracy = cv_scores.mean()
print(f"SVM Prediction Accuracy after KPCA with 10-fold cross-validation: {total_accuracy:.4f}")

## plots
plot_2d(x_kpca, sampled_y_encoded, filename = f'hw1/plots_{nsample}/kpca_mnist_firsttwo.png', title="KPCA - First Two Components - MNIST Data")


#################################
## 3. Isomap
#################################


from sklearn.manifold import Isomap

# Apply Isomap for dimensionality reduction
isomap = Isomap(n_components = 30)

## parameter tuning: number of neighbors
pipeline = Pipeline([
    ('isomap', isomap),
    ('svm', svm_model)
])

# Parameters to tune
param_grid = {
    'isomap__n_neighbors': [5, 10, 15, 20],         # Tuning the number of neighbors
}

# Using GridSearchCV to find the best parameters
grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy')
grid_search.fit(sampled_x, sampled_y_encoded)
print("Best Parameters:", grid_search.best_params_)

isomap = Isomap(n_components = 30, n_neighbors = grid_search.best_params_['isomap__n_neighbors'])

# Apply Isomap
x_isomap = isomap.fit_transform(sampled_x)

svm_model = SVC(kernel='linear', random_state=42)

# Perform 10-fold cross-validation
# cv=10 specifies 10-fold cross-validation
cv_scores = cross_val_score(svm_model, x_isomap, sampled_y_encoded, cv=10, scoring='accuracy')

# Calculate and print the total accuracy
total_accuracy = cv_scores.mean()
print(f"SVM Prediction Accuracy after Isomap with 10-fold cross-validation: {total_accuracy:.4f}")


## plots
plot_2d(x_isomap, sampled_y_encoded, filename = f'hw1/plots_{nsample}/Isomap_mnist_firsttwo.png', title="Isomap - First Two Components - MNIST Data")




#####################################
## 4. Locally Linear Embedding (LLE)
#####################################

from sklearn.manifold import LocallyLinearEmbedding

# Apply LLE for dimensionality reduction
lle = LocallyLinearEmbedding(n_components=30)

## tunning parameter for number of neighbors
pipeline = Pipeline([
    ('lle', lle),
    ('svm', svm_model)
])

# Parameters to tune
param_grid = {
    'lle__n_neighbors': [5, 10, 15, 20],         # Tuning the number of neighbors
}

# Using GridSearchCV to find the best parameters
grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy')
grid_search.fit(sampled_x, sampled_y_encoded)
print("Best Parameters:", grid_search.best_params_)

lle = LocallyLinearEmbedding(n_components = 30, n_neighbors = grid_search.best_params_['lle__n_neighbors'])

# Apply Isomap
x_lle = lle.fit_transform(sampled_x)

svm_model = SVC(kernel='linear', random_state=42)

# Perform 10-fold cross-validation
# cv=10 specifies 10-fold cross-validation
cv_scores = cross_val_score(svm_model, x_lle, sampled_y_encoded, cv=10, scoring='accuracy')

# Calculate and print the total accuracy
total_accuracy = cv_scores.mean()
print(f"SVM Prediction Accuracy after lle with 10-fold cross-validation: {total_accuracy:.4f}")


## plots
plot_2d(x_lle, sampled_y_encoded, filename = f'hw1/plots_{nsample}/LLE_mnist_firsttwo.png', title="LLE - First Two Components - MNIST Data")


#################################
## 5. Laplacian Eigenmap
#################################

from sklearn.manifold import SpectralEmbedding

# Apply Laplacian Eigenmap (spectral embedding) for dimensionality reduction
laplacian = SpectralEmbedding(n_components=30, affinity='nearest_neighbors')

n_neighbors_list = [5, 10, 15, 20]

best_score = 0
best_n_neighbors = None

for n_neighbors in n_neighbors_list:
    print(f"Testing n_neighbors = {n_neighbors}")

    # Apply Laplacian Eigenmap (Spectral Embedding) with the current n_neighbors value
    laplacian = SpectralEmbedding(n_components=30, n_neighbors=n_neighbors, affinity='nearest_neighbors')
    
    # Transform the data using Spectral Embedding
    x_laplacian = laplacian.fit_transform(sampled_x)
    
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
x_laplacian = laplacian.fit_transform(sampled_x)

svm_model = SVC(kernel='linear', random_state=42)

# Perform 10-fold cross-validation
# cv=10 specifies 10-fold cross-validation
cv_scores = cross_val_score(svm_model, x_laplacian, sampled_y_encoded, cv=10, scoring='accuracy')

# Calculate and print the total accuracy
total_accuracy = cv_scores.mean()
print(f"SVM Prediction Accuracy after Laplacian Eigenmap with 10-fold cross-validation: {total_accuracy:.4f}")

## plots
plot_2d(x_laplacian, sampled_y_encoded, filename = f'hw1/plots_{nsample}/Laplacian_mnist_firsttwo.png', title="Laplacian - First Two Components - MNIST Data")


#################################
## 6. Diffusion Map
#################################

from pydiffmap import diffusion_map as dm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Apply Diffusion Map for dimensionality reduction
diffusion_map = dm.DiffusionMap.from_sklearn(n_evecs=30)

#epsilon_list = [0.1, 1, 10]
alpha_list = [0, 0.5, 1] 

best_score = 0
best_params = None

# Create 10-fold cross-validation splits
kf = KFold(n_splits=10)
svm_model = SVC(kernel='linear', random_state=42)

# Loop through the parameter grid
for alpha in alpha_list:  # Now tuning alpha as well
    print(f"Testing alpha = {alpha}")
        
    # Apply Diffusion Map with current n_eigenpairs, epsilon, and alpha
    diffusion_map = dm.DiffusionMap.from_sklearn(n_evecs=30, epsilon='bgh', alpha=alpha)
            
    # Transform the data using Diffusion Map
    try:
        # Attempt to fit and transform the data using diffusion map
        X_diffusion = diffusion_map.fit_transform(sampled_x)
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

x_diffmap = diffusion_map.fit_transform(sampled_x)


# Perform 10-fold cross-validation
# cv=10 specifies 10-fold cross-validation
cv_scores = cross_val_score(svm_model, x_diffmap, sampled_y_encoded, cv=10, scoring='accuracy')

# Calculate and print the total accuracy
total_accuracy = cv_scores.mean()
print(f"SVM Prediction Accuracy after diffusion map with 10-fold cross-validation: {total_accuracy:.4f}")

## plots
plot_2d(x_diffmap, sampled_y_encoded, filename = f'hw1/plots_{nsample}/Diffusion_mnist_firsttwo.png', title="Diffusion Map - First Two Components - MNIST Data")



#################################
## 7. Hessian LLE
#################################


from sklearn.manifold import LocallyLinearEmbedding

# Apply Hessian LLE for dimensionality reduction
hessian_lle = LocallyLinearEmbedding(n_components=30, method='hessian', n_neighbors=10)

## tunning parameter for number of neighbors
pipeline = Pipeline([
    ('hessian_lle', hessian_lle),
    ('svm', svm_model)
])

# Parameters to tune
param_grid = {
    'hessian_lle__n_neighbors': [5, 10, 15, 20],         # Tuning the number of neighbors
}

# Using GridSearchCV to find the best parameters
grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy')
grid_search.fit(sampled_x, sampled_y_encoded)
print("Best Parameters:", grid_search.best_params_)

hessian_lle = LocallyLinearEmbedding(n_components = 30, n_neighbors = grid_search.best_params_['hessian_lle__n_neighbors'])

# Apply hessian_lle
x_hessian_lle = hessian_lle.fit_transform(sampled_x)

svm_model = SVC(kernel='linear', random_state=42)

# Perform 10-fold cross-validation
# cv=10 specifies 10-fold cross-validation
cv_scores = cross_val_score(svm_model, x_hessian_lle, sampled_y_encoded, cv=10, scoring='accuracy')

# Calculate and print the total accuracy
total_accuracy = cv_scores.mean()
print(f"SVM Prediction Accuracy after lle with 10-fold cross-validation: {total_accuracy:.4f}")


## plots
plot_2d(x_lle, sampled_y_encoded, filename = f'hw1/plots_{nsample}/Hessian_LLE_mnist_firsttwo.png', title="Hessian LLE - First Two Components - MNIST Data")

