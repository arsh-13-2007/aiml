import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

df = pd.read_csv("placement.csv")
print(df.head())
print(df.isnull().sum())
# sns.boxplot(df)   # this graph is use to detect outliers 
# plt.show()

upper_limit = df['cgpa'].quantile(0.99)
lower_limit = df['cgpa'].quantile(0.01)
print(df[(df['cgpa']>upper_limit) | (df['cgpa'] < lower_limit)])

# how ot handle outliers methods
#1. trimming
print(df.shape)
# df = df[(df['cgpa']<upper_limit) & (df['cgpa'] > lower_limit)]
# print(df.shape)
# sns.boxplot(df['cgpa'])   # it how that all outliers remove 
# plt.show()


#2. capping( winsorization )

df['cgpa'] = np.where( df['cgpa'] > upper_limit , upper_limit , 
                      np.where(df['cgpa'] < lower_limit , lower_limit , 
                            df['cgpa']
                            )
                            )
sns.boxplot(df['cgpa'])
plt.show()

