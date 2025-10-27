# what is exploratary data analysis ( EDA)
#EDA = Understanding your data before making decisions or building models.

import seaborn as sns 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


                                  

df = sns.load_dataset("tips")
print( df.head())

# # correlation with heatmap             # imp
# # it is use to represents data in 2D correlation matrix ( table) between two dimensions 
# # it calculated dependency to other means how much one depend to other 

# print(df.corr(numeric_only=True))
# sns.heatmap(df.corr(numeric_only=True))  # correlation is only calculated for int and float 
# plt.show ()




#                                     # joinplot       using seaborn 
# #jointplot() is used to visualize the relationship between two variables
# sns.jointplot( x = 'tip' , y='total_bill' , data = df  )
# plt.show()


# # there are always outliers so we learn how to find outliers in next lectures


# #  pairplot  in seaborn 

# # pairplot is also called scatter plot 
# #Multiple scatter plots for all numeric pairs
# sns.pairplot(df)
# plt.show()

# sns.pairplot(df , hue='sex')
# plt.show()


#   # distplot   plotting in seaborn
# # distplot check the disttibution of columns feature 

# #plot the distribution of a single variable

# sns.distplot(df['tip'])
# plt.show()


#   count plot       # important 
sns.countplot(x = 'sex' , data=df , color='indigo' )    # count number categary  in any column
plt.show()



#  bar plot            # important 
sns.barplot(x = 'sex' , y='tip' , data = df, color='violet' )
plt.show()