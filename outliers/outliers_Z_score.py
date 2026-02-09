import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

df = pd.read_csv("placement.csv")
print(df.head())
print(df.isnull().sum())
sns.boxplot(x = df['cgpa'] , hue=df['placed'] )
plt.show()
plt.subplot(1, 2, 1)
sns.distplot(df['cgpa'])

plt.subplot(1, 2 , 2 )
sns.distplot(df['placement_exam_marks'])

plt.show()


#     as we can see  we sue z score only for normalization curve 
#  we only use z score for cgpa 

print('mean : ' , df['cgpa'].mean())
print('std : ' , df['cgpa'].std())

# using this able to find that there exists outliers 
upper_bound =  df['cgpa'].mean()  + 3* df['cgpa'].std()
lower_bound = df['cgpa'].mean()  - 3* df['cgpa'].std()
print(df[(df['cgpa']>upper_bound ) |  (df['cgpa'] < lower_bound)])

# how to treat outliers 
#1. trimming 2.capping 
# 1. 
print( df.shape)
# df = df[(df['cgpa']<upper_bound ) &  (df['cgpa'] > lower_bound)] 
# print(df.shape)


# capping 
df['cgpa'] = np.where(df['cgpa'] > upper_bound,
                      upper_bound,
                      np.where(df['cgpa']< lower_bound,
                               lower_bound,
                               df['cgpa']
                               )
                               )


print(df['cgpa'].describe())



