import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import MinCovDet
import matplotlib.pyplot as plt

# Ensure `data` contains the following columns: 'Voltage', 'Current', 'Temperature'

import numpy as np
import pandas as pd

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

# Extract features
features = ['Voltage', 'Current', 'Temperature']
data_values = data[features].values

# Robust covariance estimation (Minimum Covariance Determinant)
mcd = MinCovDet(random_state=42).fit(data_values)
cov_matrix = mcd.covariance_
inv_cov_matrix = np.linalg.inv(cov_matrix)
mean_values = mcd.location_

# Compute Mahalanobis distances
mahalanobis_distances = np.array([
    mahalanobis(row, mean_values, inv_cov_matrix) for row in data_values
])
data['Mahalanobis Distance'] = mahalanobis_distances

# Determine anomaly threshold
threshold = np.percentile(mahalanobis_distances, 95)
data['Anomaly'] = data['Mahalanobis Distance'] > threshold

# Plot results
def plot_anomalies(data):
    plt.figure(figsize=(10, 6))
    plt.scatter(data.index, data['Mahalanobis Distance'], c=data['Anomaly'], cmap='coolwarm', label='Anomalies')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.title('Mahalanobis Distance and Anomaly Detection')
    plt.xlabel('Index')
    plt.ylabel('Mahalanobis Distance')
    plt.legend()
    plt.show()

# Plot anomalies
plot_anomalies(data)

# Display detected anomalies
anomalies = data[data['Anomaly']]
print("Detected Anomalies:")
print(anomalies)
