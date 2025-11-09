import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt

df = pd.read_csv("placement.csv")
print(df.head())
print(df.isnull().sum())
plt.show()
plt.subplot(1, 2, 1)
sns.distplot(df['cgpa'])
plt.subplot(1, 2 , 2 )
sns.distplot(df['placement_exam_marks'])
plt.show()


# placement_exam_mark are skewed 
print(df['placement_exam_marks'].describe())


sns.boxplot(x = df['placement_exam_marks']  )
plt.show()

Q1 = df['placement_exam_marks'].quantile(0.25)
Q3 = df['placement_exam_marks'].quantile(0.75)

IQR = Q3 - Q1
print( IQR)

upper_limit = Q3 + 1.5* IQR
lower_limit = Q1 - 1.5* IQR
print(upper_limit)
print(lower_limit)   # -23.5  so there is not any outlier 
# print(df[df['placement_exam_marks'] > upper_limit])  # output : outliers 


#how to handle outliers 1.tremming 
#1.  tremming  
print(df.shape)
# df =df[df['placement_exam_marks']<upper_limit]
# print(df.shape)

# sns.distplot(df['placement_exam_marks'])
# plt.show()


#2. capping 
# np.where ( condition , true , false )
new_df = df.copy()
new_df['placement_exam_marks'] = np.where ( new_df['placement_exam_marks'] > upper_limit , upper_limit , 
                   np.where( new_df['placement_exam_marks'] < lower_limit,lower_limit,
                   new_df['placement_exam_marks']
                    )
                    )
print( new_df.shape)
sns.boxplot(x = new_df['placement_exam_marks'])
plt.show()
sns.distplot(new_df['placement_exam_marks'])
plt.show()