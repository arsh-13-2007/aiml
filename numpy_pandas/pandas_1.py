# numpy and pandas very very very  to do  exploratory data analysis
# pandas is  use to analysis the data 
# what is dataframe =  it is conbination of row and  columns  and it is Tabular Structure and it is use to represent the data in excel sheet 
# to convert dictionary into dataframe
# we do pd.dataframe(dictionary_name)
import numpy as np 
import pandas as pd 

# prototype: pd.DataFrame(data=None, index=None, columns=None, dtype=None)
df = pd.DataFrame(data = np.arange(0,20).reshape(5,4), index=['row1','row2','row3','row4','row5'] ,columns=["column1","column2","column3","column4"])
print( df) 
print(df.iloc[2:4,0 :3]) # this is use top indexing in pyhton pandas 
# print( df.head(2)) # head is use to print first n line of dataframe 
# print( df.tail(3)) # tail is use to print from last n line of dataframe 
# print(df) # it use to print  full dataframe 
# df.to_csv("first_csv.csv") # this is use to convert dataframe into csv file 

# # accessing the elements 
# print(df.loc['row1']) # this is use to print particular row
# # how to check type of it
# print(type(df.loc['row1'])) # output = <class 'pandas.core.series.Series'>
# arr = np.array( df)
# print( arr.shape)
# print( arr.size)
# print( arr[0:2,0:2 ]) # in numpy we use this function to use particular indexes 
# # in pandas we use 
# print( df.iloc[0:2,0:2]) # with the help of iloc we print particular indexes in pandas
# print( type(df.iloc[0:2,0:2])) #output = <class 'pandas.core.frame.DataFrame'>


# covert dataframes into array 
print(df.iloc[:,:].values.shape) #output = arrays 
print( df["column1"].unique())
print( df.isnull().sum()) # it print  total number missing values  # very important 

# data = pd.read_csv("circuits.csv" ) # it print csv file 
# print(data)

