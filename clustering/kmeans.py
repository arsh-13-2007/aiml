import pandas as pd 
import numpy as np 
import matplotlib.pyplot  as  plt
import seaborn as sns 
from sklearn.datasets import make_blobs 

X, y = make_blobs(n_samples=1000, centers=3, n_features=2,random_state=23)
print( X.shape)