import numpy as np 
# first_list = [[1, 2, 3,4],[2,3,4,7]]
# arr = np.array(first_list)
# print(type(arr))
# print( arr.shape) # it is use to print number of row and columns in 2d array 
# arr = arr.reshape(4,2)  
# # with the help of reshape we convert m*n into n*m array 
# print(arr)


# indexing 
# how indexing is done 2d array 
arr= np.array([[1,2,3,4,5],
              [2,3,4,5,6],
              [9,7,6,8,9]])

# print(arr.size)
# print(arr.shape)
# print(  arr.reshape(5,3))
# print (arr)
# print(arr[0:2,0:2]) # important in indexing     
# print( arr [0:2, :])
# after colon it not count 


# numpy inbuild function  (important)
# arr = np.arange(0, 10  )  # it print 1D array 
# print( arr) 
# arr = np.arange( 0 , 10 , step = 2) 
# print( arr)

arr10 = np.linspace(3,10 , 10,  dtype = int   ).reshape( 2,5)# in this example 1 is starting point and 2 is ending point 
print( arr10)
# print( arr.shape)

# # copy of this array 
# arr1 = arr
# arr1[3:] = 500 # it use to replace all elements by 500 after 3 index 
# print(arr1) # this make change it original arr also so to overcome this we use copy function 

# arr1 =arr.copy()
# arr1[3:] = 1000
# print(arr1) # copyed array after some 
# print(arr) # original array not affected

# arr = np.array([[1,2,3,4],[2,3,0,5],[1,6,7,8]])
# print(arr)
# print(arr.size)
# print(arr.shape)
# print( arr[0:3,2:4]) # first show row and second show columns 


# condition very usefull in exploratory data analysis 
# val = 2 
# print ( arr < val) # it return boolen answer  if correct then return true else return false 
# # basic operation of array
# print( arr*2 )
# print( arr-2 )
# print( arr/ 2 ) 

# important function of array in numpy 
# first 
# create and reshape the array
arr = np.arange(0, 10 ).reshape(2,5)  # we should remind that it size is not greater the given size in arrange function it always equal to this size
# print( arr)
arr1 = np.arange( 0, 20 , step = 2).reshape(2, 5)
print( arr* arr1)
# function or array   
arr = np.ones((2,5),dtype=int) # dtype keyword is use to give datatype 
print(arr)


# use of random function 

arr = np.random.rand(2,2) #it always select  random number between 0 and 1 
print(arr)

arr = np.random.randn( 3,3 ) # it do standard distribution 
print( arr)

arr = np.random.randint( low = 4, high= 6, size = (3, 3))  # random integer between 4 and 6  remind  that 6 excluded
print(arr)



 