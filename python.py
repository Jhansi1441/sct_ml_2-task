import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Generate a random dataset for customers' purchase history
np.random.seed(42)  # For reproducibility

n_customers = 200  # Number of customers

# Features for customer purchase history
# Random values to simulate: total_spending, avg_basket_size, num_transactions
total_spending = np.random.uniform(100, 5000, size=n_customers)
avg_basket_size = np.random.uniform(20, 300, size=n_customers)
num_transactions = np.random.uniform(1, 100, size=n_customers)

# Create a DataFrame for the dataset
data = pd.DataFrame({
    'Total_Spending': total_spending,
    'Avg_Basket_Size': avg_basket_size,
    'Num_Transactions': num_transactions
})

# Step 2: Standardize the data (important for K-Means)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Step 3: Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data_scaled)

# Step 4: Add the cluster labels to the dataset
data['Cluster'] = kmeans.labels_

# Step 5: Visualize the clusters
plt.figure(figsize=(10, 6))

plt.scatter(data['Total_Spending'], data['Avg_Basket_Size'], c=data['Cluster'], cmap='viridis', marker='o', s=50)
plt.title("Customer Segmentation based on Purchase History")
plt.xlabel("Total Spending")
plt.ylabel("Average Basket Size")
plt.colorbar(label='Cluster')

plt.show()

# Display the first few rows of the dataset with cluster labels
data.head()
