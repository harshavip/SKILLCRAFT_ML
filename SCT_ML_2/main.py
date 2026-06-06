import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Load dataset
data = pd.read_csv("Mall_Customers.csv")

print("Dataset loaded successfully")
print("\nFirst 5 rows:")
print(data.head())

print("\nDataset information:")
print(data.info())

print("\nMissing values:")
print(data.isnull().sum())


# Select features for clustering
# Annual Income and Spending Score are used to group customers
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

print("\nSelected features:")
print(X.head())


# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Find best number of clusters using Elbow Method
wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)


# Plot Elbow Method graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method to Find Best K")
plt.grid(True)
plt.show()


# Apply K-Means clustering
# For Mall Customer dataset, commonly k = 5
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
data["Cluster"] = kmeans.fit_predict(X_scaled)


print("\nCustomer data with cluster numbers:")
print(data.head())

print("\nNumber of customers in each cluster:")
print(data["Cluster"].value_counts())


# Plot customer clusters
plt.figure(figsize=(8, 6))

plt.scatter(
    data["Annual Income (k$)"],
    data["Spending Score (1-100)"],
    c=data["Cluster"],
    cmap="viridis",
    s=60
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using K-Means Clustering")
plt.colorbar(label="Cluster")
plt.grid(True)
plt.show()