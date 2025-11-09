# bivariate  / multivariate analysis 
import pandas as pd 
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt 

titanic = pd.read_csv("Titanic.csv")
flight = sns.load_dataset('flights')
iris = sns.load_dataset('iris')
tips = sns.load_dataset('tips')

print (titanic.head())
print (iris.head())
print (flight.head())
print (tips.head())

# sns.heatmap (iris.corr(numeric_only=True) , annot = True )
# plt.show()
 
sns.heatmap (tips.corr(numeric_only=True) , annot = True )
plt.show()

sns.scatterplot(x= tips['tip'] , y=tips['total_bill'] , hue=tips['sex'] , style=tips['smoker']) # numercial - numercial  multivariate analysis 
plt.show() 

sns.barplot(x = titanic['Pclass'] , y = titanic['Age'] , hue=titanic['Sex'])
plt.show()

sns.boxplot( x = titanic['Sex'] , y = titanic['Age'] , hue=titanic['Survived'])
plt.show()

sns.distplot(titanic[titanic['Survived'] == 0]['Age'] )
sns.distplot(titanic[titanic['Survived'] == 1]['Age'] )
plt.show()