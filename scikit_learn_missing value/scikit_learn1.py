import pandas as pd 
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

df = fetch_openml('Titanic' , version= 1 , as_frame=True)['data']
# print( df.info())       
# print(df.isnull().sum())
# use graph for visualization
# sns.heatmap(df.isnull(), yticklabels=False )
# plt.show()
# print(df.shape)
# print(df.head())

# # this is way to  handle the null values by deleting the whole column  which is not perferrable  so to  handle missing values we use imputers 
df.drop(['body' ,'boat'] , axis= 1 , inplace= True)  # axis = 1 show that it delete/ drop  column if axis = 0 then it drop/ delete row 
print(df.shape)
sns.heatmap(df.isnull(), yticklabels=False )
plt.show()
"""               imputer =  use to handle missing values   it use technique like - mean , median , mode          """

print(df['age'].isnull().sum())

# firstly we using means   # for numercial columns only 
imp =  SimpleImputer(strategy= 'mean')    # it replace  mean value inplace if null value  in age column 
df['age'] = imp.fit_transform(df[['age']])
print(df['age'].isnull().sum())

sns.heatmap(df.isnull(), yticklabels=False )
plt.show()

# df = pd.get_dummies(df, columns=['sex'])
df['sex'] = df['sex'].map({'male': 1, 'female': 0})


print( df.head())
df = df[['sex']]
print( df )