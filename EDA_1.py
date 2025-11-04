

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
#                                      problem statement is = passenger is survived  or not 
data = pd.read_csv('titanic.csv')
# print(data)
# print(data.isnull().sum())

print( data.isnull()) 

# concept we use visualization to find null value
sns.heatmap(data.isnull()  , cbar=False , yticklabels=False )
plt.show() 

sns.countplot(x='Survived' , data = data )
plt.show() 

"""use of hue """
#the hue parameter in Seaborn is one of the most powerful and visually useful features!


sns.countplot(x='Survived' , hue='Sex' , data= data)
plt.show()

sns.countplot(x='Survived' , hue='Pclass' , data = data )
plt.show()


#     dropna() is a method used with Pandas DataFrames or Series to drop (remove) rows or columns that contain NaN

sns.distplot(data['Age'].dropna() ,bins=40 )  
plt.show() 

       # main target 
      # firslty we learn how to handle missing value inside dataset

sns.boxplot(x='Pclass' , y='Age' ,data=data )   # using box we able to find average values of age and pclass 
plt.show()


def function ( cols):
    Age = cols[0]
    Pclass = cols[1]
    if data.isnull(Age):
        if Pclass == 1:
            return  37 
        elif Pclass == 2:
            return 29
        elif Pclass == 3:
            return 24
    else: 
        return Age
