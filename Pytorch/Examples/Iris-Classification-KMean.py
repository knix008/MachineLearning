import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from kmeans_pytorch import kmeans, kmeans_predict
import matplotlib.pyplot as plt

df = pd.read_csv('data/iris.csv') 
df.info()
print('------------------------------------------')
print(df)

data=pd.get_dummies(df, columns=['Species'])
print(data)

from sklearn.model_selection import train_test_split
x, y = train_test_split(data, test_size=0.2, random_state=123)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
    
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit(data).transform(x)
y_scaled = scaler.fit(y).transform(y)

x = torch.from_numpy(X_scaled)
y = torch.from_numpy(y_scaled)

print(x.size())
print(y.size())
print(x)

num_clusters = 3
cluster_ids_x, cluster_centers = kmeans(
    X=x, num_clusters=num_clusters, distance='euclidean', device=device
)

print(cluster_ids_x)
print(cluster_centers)

cluster_ids_y = kmeans_predict(
    y, cluster_centers, 'euclidean', device=device
)

print(cluster_ids_y)

plt.figure(figsize=(4, 3), dpi=160)
plt.scatter(y[:, 0], y[:, 1],c=cluster_ids_y, cmap='viridis', marker='x')

plt.scatter(
    cluster_centers[:, 0], cluster_centers[:, 1],
    c='white',
    alpha=0.6,
    edgecolors='black',
    linewidths=2
)

plt.tight_layout()
plt.show()