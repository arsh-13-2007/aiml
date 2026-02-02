import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import datetime 

date = pd.read_csv("orders.csv")
time = pd.read_csv("messages.csv")

# print(data.head())
# print(time.head())
# print(data.isnull().sum())
# print(time.isnull().sum())

print( date.info())
print( time.info())

# first is to change the dtype of data column  to datatime64 

date['date'] = pd.to_datetime(date['date'])
print(date.info())

#  how to extract year 

date['date_year'] = date['date'].dt.year

print(date.head())

#  how to extract month_number

date['date_month_no'] = date['date'].dt.month

print(date.head())

#  how to extract month_name 

date['date_month_name'] = date['date'].dt.month_name()

print(date.head())

#  how to extract day

date['date_day'] = date['date'].dt.day

print(date.head())

# how to extract day of week 

date['date_dow'] = date['date'].dt.dayofweek
print(date.head())

# how to extract day of week-name 

date['date_dow_name'] = date['date'].dt.day_name()

print(date.head())


# is weekend ? 


date['date_is_weekend'] = np.where(date['date_dow_name'].isin(['Sunday' , 'Saturday']),1,0)


print(date.head())

# # how to extract week_number of the year 

date['date_week'] = date['date'].dt.isocalendar().week

print(date.head())


# how to extract quarter 

date['quarter'] = date['date'].dt.quarter

print(date.head())

# extract semester 


date['semester'] = np.where(date['quarter'].isin([1,2]),1,2)


print(date.head())

# extract time elapsed between dates 

today = datetime.datetime.today()

# print( today)

# print(today-date['date'])

# print(today-date['date'].dt.days)


#                     time 

print(time.info())
time['date'] = pd.to_datetime(time['date'])



time['hour'] = time['date'].dt.hour
time['min'] = time['date'].dt.minute
time['second'] = time['date'].dt.second

print(time.head())


print(today-time['date'])







