# what is exploratary data analysis ( EDA)
#EDA = Understanding your data before making decisions or building models.

import seaborn as sns 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt



# distplot   plotting in seaborn

df = sns.load_dataset("tips")
print( df.head())

# correlation with heatmap             # imp
# it is use to represents data in 2D correlation matrix ( table) between two dimensions 
# it calculated dependency to other means how much one depend to other 

print(df.corr(numeric_only=True))
sns.heatmap(df.corr(numeric_only=True))  # correlation is only calculated for int and float 
plt.show ()

