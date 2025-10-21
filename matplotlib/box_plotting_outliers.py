import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
# box plot 
# box plot is Helps in detecting outliers easily.

"""step of removing outliers 
  It shows the minimum, first quartile (Q1), median (Q2), third quartile (Q3), and maximum values.
  
IOR =  Q3 - Q1 

  lower = Q1 - 1.5* ( IQR)
  upper = O3 + 1.5*( IQR)
  
mainly (Q1 is 25 %)
 (Q3 is 75 % )
 IQR = Q3 - Q1 
""" 

"""" example """

data = [11,10,12,14,12,15,14,13,15,102,12,14,17,19,107,10,13,12,14,12,108,12,11,14,13,15,10,15,12,10,14,13,15,100]
# plt.hist( data , color = 'r' , linestyle='dotted' , linewidth=2 , bins=None )
# plt.title('learning')
# plt.xlabel('x axis')
# plt.ylabel('y axis')
# plt.show()

# finding outliers # using IQR techinque 
# sort the data firstly 
data= sorted(data )  # sorted function is to sort the list or any thing 
# print(data)
Q1 , Q3 = np.percentile(data , [25 , 75 ])
# print( Q1 , Q3)
IQR = Q3 - Q1 
lower_fence = Q1 - 1.5*(IQR)
upper_fence = Q3+ 1.5*(IQR)
# print( lower_fence , upper_fence)


plt.boxplot(data , vert = True ,  patch_artist= True  ) # dot give outliers 
plt.show()
