## BIOS774 HW1 Benchmarking Spectral Methods
# Author: Xinyu Zhang
# Date: Sept. 28th

# Import Packages
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Import data
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#################################
## 0. Data Preprocessing
#################################

# Reshape the data: Flatten the 28x28 images into 1D arrays of 784 pixels
x_train_flattened = x_train.reshape(x_train.shape[0], -1)
x_test_flattened = x_test.reshape(x_test.shape[0], -1)

## change into pandas format
x_train_flattened = pd.DataFrame(x_train_flattened)
x_test_flattened = pd.DataFrame(x_test_flattened)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

# Combine x_train and x_test into one dataframe
x = pd.concat([x_train_flattened, x_test_flattened], axis=0).reset_index(drop=True)

# Combine y_train and y_test into one dataframe
y = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)

# Merge x and y into one dataframe for sampling
data = pd.concat([x, y], axis=1)
data.columns = data.columns.astype(str)
data.columns.values[-1] = 'y'
x_columns = data.columns[:-1]  # All columns except the last one

## 1. downsample to a sample size of 500
sampled_data_500 = data.groupby("y").apply(lambda x: x.sample(n=50, random_state=12)).reset_index(drop=True)

X_500 = sampled_data_500[x_columns]  # Extract x columns (features)
y_500 = sampled_data_500['y']   # Extract y column (target)

scaler = StandardScaler()
X_500_standardized = scaler.fit_transform(X_500)  # Standardize the x values

sampled_data_500_standardized = pd.DataFrame(X_500_standardized, columns=x_columns)  # Recreate the DataFrame for x
sampled_data_500_standardized['y'] = y.reset_index(drop=True)  # Add the y column back

## 1. downsample to a sample size of 1000
sampled_data_1000 = data.groupby("y").apply(lambda x: x.sample(n=100, random_state=12)).reset_index(drop=True)

X_1000 = sampled_data_1000[x_columns]  # Extract x columns (features)
y_1000 = sampled_data_1000['y']   # Extract y column (target)


scaler = StandardScaler()
X_1000_standardized = scaler.fit_transform(X_1000)  # Standardize the x values

sampled_data_1000_standardized = pd.DataFrame(X_1000_standardized, columns=x_columns)  # Recreate the DataFrame for x
sampled_data_1000_standardized['y'] = y.reset_index(drop=True)  # Add the y column back

## 2. downsample to a sample size of 5000
sampled_data_5000 = data.groupby("y").apply(lambda x: x.sample(n=500, random_state=12)).reset_index(drop=True)

X_5000 = sampled_data_5000[x_columns]  # Extract x columns (features)
y_5000 = sampled_data_5000['y']   # Extract y column (target)

scaler = StandardScaler()
X_5000_standardized = scaler.fit_transform(X_5000)  # Standardize the x values

sampled_data_5000_standardized = pd.DataFrame(X_5000_standardized, columns=x_columns)  # Recreate the DataFrame for x
sampled_data_5000_standardized['y'] = y.reset_index(drop=True)  # Add the y column back



## 3. downsample to a sample size of 10000
sampled_data_10000 = data.groupby("y").apply(lambda x: x.sample(n=1000, random_state=12)).reset_index(drop=True)

X_10000 = sampled_data_10000[x_columns]  # Extract x columns (features)
y_10000 = sampled_data_10000['y']   # Extract y column (target)

scaler = StandardScaler()
X_10000_standardized = scaler.fit_transform(X_10000)  # Standardize the x values

sampled_data_10000_standardized = pd.DataFrame(X_10000_standardized, columns=x_columns)  # Recreate the DataFrame for x
sampled_data_10000_standardized['y'] = y.reset_index(drop=True)  # Add the y column back


# Sample 500 data points for each class in y
sampled_data_500_standardized.to_csv('hw1/data/sample_data_500_standardized.csv', index=False)
sampled_data_1000_standardized.to_csv('hw1/data/sample_data_1000_standardized.csv', index=False)
sampled_data_5000_standardized.to_csv('hw1/data/sample_data_5000_standardized.csv', index=False)
sampled_data_10000_standardized.to_csv('hw1/data/sample_data_10000_standardized.csv', index=False)

sampled_data_500.to_csv('hw1/data/sample_data_500.csv', index=False)
sampled_data_1000.to_csv('hw1/data/sample_data_1000.csv', index=False)
sampled_data_5000.to_csv('hw1/data/sample_data_5000.csv', index=False)
sampled_data_10000.to_csv('hw1/data/sample_data_10000.csv', index=False)