import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

data = pd.read_csv("Titanic.csv")
# print(data.head(15))
# print(data.isnull().sum())
# sns.heatmap(data.isnull())
# plt.show()
data =  data[['Survived' ,'Pclass' , 'Sex' ,'Age' , 'SibSp']]   # we select main columns and row that is in use 
print( data.head())