# supervised  learning is used labelled data 
# unupervised learning is used unlabelled data then learn pattern and make  prediction 
# semi supervised learning is used both labelled and unlablled dataset for eg : self-training , co-training
# reinforcement learning                             for eg: Q-learning ,  and many more   

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler , MinMaxScaler , OneHotEncoder , LabelEncoder            
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score ,f1_score , accuracy_score , precision_score

df = pd.read_csv("Titanic.csv")
print( df.head())
print(df.isnull().sum())
print(df.duplicated().sum())
print(type(df))
print(df.shape)
print(df.size)