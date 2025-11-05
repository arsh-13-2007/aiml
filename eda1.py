import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Titanic.csv")
print( data.head ())

sns.countplot(x='Pclass', data=data)  # graph is use for categorical data
plt.show()


value = data['Pclass'].value_counts().sort_index()

plt.pie(value, labels=['one', 'two', 'third'], autopct="%0.2f")
plt.show()

