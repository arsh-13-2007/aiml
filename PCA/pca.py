import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.datasets  import load_breast_cancer
import numpy as np 
from sklearn.decomposition import PCA
# from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import StandardScaler
scale = StandardScaler() 


cancer = load_breast_cancer()   # load dataset from sklearn 

data  = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])   # use only feature of the dataset that i load from sklearn  
# print( data)
# data_scaled= minmax_scale(data)  
data_scaled= scale.fit_transform(data)
print(data_scaled)


print( "after reduction using pca ")

pca = PCA(n_components=2)  # i reduce  30 dimemsion into 2 dimensions using pca     

data_pca = pca.fit_transform(data_scaled)
print( data_pca)  #  WHOLE 30 DIMENSION  conbined into only 2 feature 
print( data_pca.shape ) # output (569, 2)
print( data_scaled.shape) # output (569, 30)


print(pca.explained_variance_)

plt.scatter(data_pca[:,0] , data_pca[: , 1 ], c=cancer['target'])
plt.xlabel('first pca')
plt.ylabel('second pca')
plt.show()  