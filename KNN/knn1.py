# knn is for labled dataset 

import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
from  sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split 


X, y = load_wine(return_X_y=True )
