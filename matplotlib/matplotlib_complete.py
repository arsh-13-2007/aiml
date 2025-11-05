import numpy as np
import pandas as pd 
from io import StringIO
import matplotlib.pyplot as plt # it is use to low level data Virtualization  for advance data virtualization we use seaborn 

x = np.arange(0  ,10)
# y = np.arange( 11 , 21 )

# scatter   plot 
y = x*x 
# print(plt.scatter( x, y, c='b'))
# plt.title("learning")
# plt.xlabel("x-axis")
# plt.ylabel("y-axis")
# # plt.savefig("test.png") # this is use to save this plot in computer

# plt.plot(x, y, linewidth=2, color='r', marker='o', linestyle='--')# this is use to draw line through points 

# # in linestyle we give may pattern like (-','--','-.',':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted )
# plt.show()  # to print we use show command 


# we do subplotting also 
# subplot means in frame we able to show multiple graph 
 
# plt.subplot(2, 2,1 ) # parameter we required : ( nrow , ncolumns , position( index)) 
# plt.plot(x, y, c='b' , linestyle = 'dashdot')
# plt.subplot( 2, 2, 2)
# plt.plot( x, y, linestyle='dotted', linewidth= 2 , c = 'r' )
# plt.subplot( 2, 2, 3)
# plt.plot( y , x , linestyle = 'dashed')
# # plt.show() 

# let solve differnt question  ( important )
y = 3*x + 5 
# plt.plot( x, y  , linestyle= "dashdot" , linewidth= 2 , c= 'brown')
# plt.show ()   


#  sin form graph 
# print(np.pi )  # output =  3.141592653589793
# x= np.arange ( 0 , 4*np.pi , step =  0.1)
# y = np.sin(x) 

# plt.plot( x, y , linestyle="dashed" , linewidth = 2 , c = 'r') # very very very important 
# plt.show()

# subplot in  sin _cos  form graph
# x = np.arange( 0 , 5*np.pi , step = 0.1)
# y_sin = np.sin(x)
# y_cos = np.cos(x)


# plt.subplot(2, 2,1 ) # parameter we required : ( nrow , ncolumns , position( index)) 
# plt.plot(x, y_sin, c='b' , linestyle = 'dashdot')
# plt.title( "sin plot")
# plt.subplot( 2, 2, 2)
# plt.plot( x, y_cos, linestyle='dotted', linewidth= 2 , c = 'r' )
# plt.title("cos plot")
# plt.subplot( 2, 2, 3)

# plt.plot( y_sin , y_cos , linestyle = 'dashed')
# plt.title("mix plot ")
# plt.show() 


# bar plot 


# x = [ 2, 8, 10]
# y= [ 11,16, 9]

# x1 = [ 3, 9, 11]
# y1 = [ 6, 15, 7]
# plt.bar( x, y , color='r')
# plt.bar( x1, y1 , color='green')
# plt.title("bar plotting")
# plt.xlabel('x-axis')
# plt.ylabel('y-axis')
# plt.show()


# histogram plot 

# a = np.array([22,87,65,43,23,98,65,1,55,0,45])
# plt.hist( a , color='green' , linewidth=2 , linestyle = 'dotted' , bins= 10) # bins is use to give gave between that range 
# plt.title("histogram plotting")
# plt.xlabel('x-axis')    # x -axis represents in histogram ( given data) 
# plt.ylabel('y-axis')   # in histogram y axis represents the density means ( how many number occur between that paticular range)
# plt.show()


#  box plot using matplotlib

# data = [np.random.normal( 0 , std ,  100 ) for std in range ( 1, 4)]

# print( data )
# plt.boxplot( data , vert= True, patch_artist= True )  # vert = vertical 
# plt.show() 


# pie chart 


labels = ['python', 'c++', 'java', 'c']
values = [215, 130, 245, 210]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.4, 0, 0.2, 0)

plt.pie(values, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True)
plt.axis('equal')
plt.show()
