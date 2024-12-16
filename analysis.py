import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
import matplotlib.pyplot as plt


# Ensure `data` contains the following columns: 'Voltage', 'Current', 'Temperature'



# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data
n_samples = 500

# Normal data (Cluster 1)
voltage_1 = np.random.normal(loc=28.0, scale=0.5, size=n_samples)  # Voltage
current_1 = np.random.normal(loc=3.0, scale=0.2, size=n_samples)   # Current
temperature_1 = np.random.normal(loc=50.0, scale=2.0, size=n_samples)  # Temperature

# Normal data (Cluster 2 - slightly different operating range)
voltage_2 = np.random.normal(loc=29.5, scale=0.3, size=n_samples)  # Voltage
current_2 = np.random.normal(loc=2.8, scale=0.1, size=n_samples)   # Current
temperature_2 = np.random.normal(loc=45.0, scale=1.5, size=n_samples)  # Temperature

# Concatenate clusters into one dataset
voltage = np.concatenate([voltage_1, voltage_2])
current = np.concatenate([current_1, current_2])
temperature = np.concatenate([temperature_1, temperature_2])

# Introduce some anomalies
n_anomalies = 20
voltage_anomalies = np.random.uniform(low=26.0, high=31.0, size=n_anomalies)
current_anomalies = np.random.uniform(low=1.0, high=4.0, size=n_anomalies)
temperature_anomalies = np.random.uniform(low=30.0, high=70.0, size=n_anomalies)

voltage = np.concatenate([voltage, voltage_anomalies])
current = np.concatenate([current, current_anomalies])
temperature = np.concatenate([temperature, temperature_anomalies])

# Create a DataFrame
data = pd.DataFrame({
    'Voltage': voltage,
    'Current': current,
    'Temperature': temperature
})

# Shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

# Display the dataset
data.head()

# Assuming `data` is the dataset with columns ['Voltage', 'Current', 'Temperature']
features = ['Voltage', 'Current', 'Temperature']
data_values = data[features].values

# Compute the mean vector and covariance matrix
mean_vector = np.mean(data_values, axis=0)
cov_matrix = np.cov(data_values, rowvar=False)
inv_cov_matrix = np.linalg.inv(cov_matrix)

# Calculate Mahalanobis Distance for each data point
mahalanobis_distances = np.array([
    mahalanobis(row, mean_vector, inv_cov_matrix) for row in data_values
])

# Compute the squared Mahalanobis distance
squared_mahalanobis_distances = mahalanobis_distances ** 2

# Define significance level and degrees of freedom
alpha = 0.01  # 1% significance level
degrees_of_freedom = len(features)

# Compute the Chi-Squared critical value
chi_squared_threshold = chi2.ppf(1 - alpha, degrees_of_freedom)

# Flag outliers
data['Squared Mahalanobis Distance'] = squared_mahalanobis_distances
data['Chi-Squared Threshold'] = chi_squared_threshold
data['Outlier'] = squared_mahalanobis_distances > chi_squared_threshold

# Display results
outliers = data[data['Outlier']]
print("Detected Outliers:")
print(outliers)

# Visualization (optional)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(data.index, squared_mahalanobis_distances, label="Data Points")
plt.axhline(y=chi_squared_threshold, color='r', linestyle='--', label="Chi-Squared Threshold")
plt.title("Chi-Squared Outlier Detection")
plt.xlabel("Index")
plt.ylabel("Squared Mahalanobis Distance")
plt.legend()
plt.show()
