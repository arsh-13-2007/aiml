import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.cluster import  KMeans



X , y = make_blobs(n_samples=1000 , centers=3 , n_features=2 , random_state=23)
plt.scatter(X[: , 0] , X[: , 1]   )
plt.show()

X_train, X_test = train_test_split( X,  test_size=0.33, random_state=42)

# manual way to finding k

wcss =[]
for k in range ( 1 , 11):
    kmeans = kmeans(n_cluster=k , init='k-means++')
    kmeans.fit(X_train)
    wcss.append(kmeans.inertia_)
