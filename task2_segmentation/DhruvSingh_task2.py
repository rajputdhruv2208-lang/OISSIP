import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Dataset load  
df = pd.read_csv('Mall_Customers.csv')

# 2. columns (Annual Income aur Spending Score)
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# 3. K-Means model identify 
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# 4. plot the result
plt.figure(figsize=(10, 6))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=df['Cluster'], cmap='viridis')
plt.title('Customer Segmentation')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()