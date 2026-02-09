import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
df = pd.read_csv("data_science_job.csv")
print (df.isnull().mean()*100)

print( df.shape)
<<<<<<< HEAD
# df.drop(['experience'] , axis=  1  , inplace = True )   # we want ot drop any column 
#  we apply cca only for those who have missing value smaller then 5% 
df = df.dropna(subset= ['experience' ,'education_level' , 'enrolled_university' , 'training_hours'])  # if we want ot drop row from any particualar column 
=======
# df.drop(['experience'] , axis=  1  , inplace = True )
df = df.dropna(subset= ['experience' ,'education_level' , 'enrolled_university' , 'training_hours'])  # if we want to drop row from any particualar column 
>>>>>>> 02c00d4965c420214249fab964c67505d55fcffc
# df = df.dropna() # we we want to drop all row having missing value 
print(df.head())
print( df.shape) 