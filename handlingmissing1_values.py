import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
df = pd.read_csv("data_science_job.csv")
print (df.isnull().mean()*100)

#  we apply cca only for those who have missing value smaller then 5% 
print( df.shape)
# df.drop(['experience'] , axis=  1  , inplace = True )
df = df.dropna(subset= ['experience' ,'education_level' , 'enrolled_university' , 'training_hours'])  # if we want to drop row from any particualar column 
# df = df.dropna() # we we want to drop all row having missing value 
print(df.head())
print( df.shape) 