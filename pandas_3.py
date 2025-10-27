import pandas as  pd  
import numpy as np
from io import StringIO , BytesIO
import json
import pickle

# numpy previous lecture revision :- 

arr = np.arange( 0 , 20 , step = 2 ).reshape(2, 5 )
# print( arr )
# print( arr.size)
# print( arr.shape)
# print( arr[0:1,2:4])
# arr1 = np.copy(arr) # copy function 
# print(arr1) 
# print( arr1 > 12)

# need to revise again 


# print( arr[0 : 2 , 3: 5 ])  # first represents rows and second represents columns 
# arr2  = np.random.rand(3, 3) # random number between 0 and 1 ṇ
# print( arr2)
# arr2 = np.random.randint( 2 ,20 , ( 3, 4))
# print(arr2)
# arr2 = np.random.randn(3,3) # it do standard distribution 
# print( arr2)
# arr2 = np.ones((3,3 ), dtype = int  ) # default datatype = float 
# print(arr2)
# arr5 = np.linspace( 2, 20  , 10 , dtype= int ).reshape( 2, 5)
# print( arr5)

# pandas_ 1 lecture revision 
df = pd.DataFrame(np.arange(2, 20).reshape(3,6) , index=( 'row1','row2','row3'), columns = ( "column1", "column2", "column3" , "column4","column5", "column6"))
# print( df ) 
# print( type( df)) #output = <class 'pandas.core.frame.DataFrame'>

# print( df.to_csv("test1.csv")) # csv  = coma separtate file 
# print(df.iloc[0:2,3:6 ])  # iloc means index location 
# print( df.head(3))
# print( df . tail( 2))
# data = pd.read_csv("second.csv")
# print ( data.isnull().sum())
# print(np.nan) # use to give null value in numpy in python 
# need to revise again 

# print ( df.loc[:,:].values) # it convert it into array                                             # important 

# print( df.loc['row2'])

# loc is use to print particular row you can check in above example also 


# print(df["column2"].unique)
print(df["column2"].value_count())

# pandas_2 lecture revision

# data = pd.read_csv("test.csv")
# print(data)
# print( data . info())   
# print (data.describe())
# print( data["col2" ])
# print( data["col2" ][2])
# data =('col1,col2,col3\n'
#        'x,y,1\n'
#        'a,b,2\n'
#        'c,d,3')
# print(type( data ))
# df = pd.read_csv(StringIO(data))
# print( pd.read_csv( StringIO( data) ,usecols=["col1"]))







# lecture 3 pandas 

"""A JSON file contains data in a JavaScript Object Notation format, 
which is a human-readable, text-based standard used to store and exchange structured data
 between applications, servers, and web pages"""

# Data = '[{ "employee_name": "arsh", "email": "arsh@gmail.com", "age": 18, "phone_number": 741852963 }]'
# df = pd.read_json(Data)
# print(df)
"""json is very improtant """
"""  learn json from krish naik youtube video link -> https://youtu.be/GWUGFjdUO7w"""

# json url to csv 
# df = pd.read_csv( 'url link of json file' , header = None ) 
# print( df )

# data = df.to_csv("json.csv")
# data = pd.read_csv("json.csv ")
# print( data.head())
# print( data.tail())
# print(data.isnull().sum())
# print(data["age"][0])
# print( data.loc[:,:].values) # convert to it into array 
# print( data.info())
# print( data.describe())

# url to csv 
# df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data' , header = None)
# print( type(df) )   # output  <class 'pandas.core.frame.DataFrame'>

# df.to_csv("json2.csv")
# data = pd.read_csv("json2.csv")
 

# print(data.to_json( orient='records'))  # important  very very in every time  ut store data in record fromat
# print( data.to_read( orient = 'index ') ) # it print or store in index format 


# IF WE WANT TO CONVERT ANY DATAFRAME INTO JSON 
# (DF) IS  ANY DATAFRAME
# PRINT(DF.TO_JSON())


# TO  CONVERT ALL DATA INTO NORMAL FROM MEANS CONVERT NESTED DATA  ALSO INTO NORMAL FROM OF JSON 
# data = in json format or in list form 
#  PRINT(pd.json_normalize(DATA)) 


# # reading html content 

# url = 'https://www.fdic.gov/bank/individual/failed/banklist.html'
# from_html = pd.read_html(url)
# print( type(from_html))
# print(from_html[0])

# convert any data into in html page 
# from_html.to_html("demo.html")


# reading excel file 

# df_excel = pd.read_excel('excel_sheet.xlsx ')
# # print(df_excel.head())

# # reading xml file 
# """An XML file is a document written in Extensible Markup Language (XML),
# a markup language designed to store and transport data"""


# # pickling 
# # ( very very important in point of machine learning  when we start makeing machine learning model )
# """Pickling → Process of converting a Python object into a byte stream (so it can be saved to a file or sent over a network).

# Unpickling → Process of converting the byte stream back into the original Python object.""" 

# df_excel.to_pickle('df_excel')  # it is use to save the data to disk 
# data = pd.read_pickle('df_excel')
# print( data )


# import pickle

# # Pickling
# data = {"name": "Arsh", "age": 21}
# with open("data.pkl", "wb") as f:
#     pickle.dump(data, f) 
# # Unpickling
# with open("data.pkl", "rb") as f:
#     loaded_data = pickle.load(f)

# print(loaded_data)  # {'name': 'Arsh', 'age': 21}
