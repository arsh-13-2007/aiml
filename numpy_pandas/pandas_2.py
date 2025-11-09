# ml is subset of an ai( artificial intelligence )

import pandas as pd 
import numpy as np
from io import StringIO
#   revise pandas commands which i learn in previous lecture 
# important commmand or function which i learn in previous lecture 
# very very important 

# df = pd.DataFrame(np.arange(10,30 ).reshape(5,4), index = ( 'row1' ,'row2' ,'row3' ,'row4' ,'row5' ), columns=("column1","column2","column3","column4"))
# print( df)
# df.to_csv("second.csv")
# print( df.head(3))
# print( df.tail(2))
# print( df.iloc[0:2,2:5]) 
# print( df.loc[:,:].values)        # use to convert df into array 
# print( df.loc['row1'])
# print(df["column2"].unique)
# print( df.isnull().sum())


# starting learn next lecture 
# full form of csv is coma seprated values 
df = pd.read_csv( "circuits.csv")
print( df.info()) # it is use to print all inforamtion about csv file 
print( df.describe()) # very very important output = max , min, std, mean , 25% etc. but only for integer float etc. not for string 
print( df["country"].value_counts()) # in this we enter only any column of csv file # and give count as output 
print( df[df['info']> 100]) # in it also you enter column name  and which condition you want 
 

 #  from string to csv 


data =('col1,col2,col3\n'
       'x,y,1\n'
       'a,b,2\n'
       'c,d,3')
print(type(data))
df = pd.read_csv(StringIO(data))
print(df)
print( pd.read_csv(StringIO(data),usecols=['col1', 'col3'])) # if we want to read specify columns so we use this function 

# if we want to convert string in csv file 
df.to_csv('test.csv')

print(" ")
df = pd.read_csv(StringIO(data) , dtype= object  ) # object means it return string type  answer # we can covert it into any from like int or float etc. 

print(  df['col1']) # it print whole row 

print("  ")
print( df['col1'][1]) # it print particular row and column

# we can give different datatype to each column also  ( important )


df = pd.read_csv( StringIO(data) , dtype={'col1':object, 'col2' : object , 'col3':float} , index_col= 0 ) #important

#  if we want to shift the index another column then  we use (index_col = any_column_index ) important 
print(df) # print without index 
df = pd.read_csv( StringIO(data) , index_col=False)
print( df)


data = ' a,b\n"hello,\\"bob\\",nice to see you",5'
df = pd.read_csv(StringIO(data) , escapechar='\\')  # escapechar is use to remove any char in string or csv we use it 
print(df)



# how to covert url to csv  file 




# df = pd.read_csv( " any link you want to paste" , sep='\t')   # very very important if we want to read any csv through link 



