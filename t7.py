import numpy as np
import cv2
import matplotlib.pyplot as plt

#Q1
array_book1 = np.loadtxt("book1.csv",delimiter = '\t',usecols = 1,skiprows = 1)
max = array_book1.max()
min = array_book1.min()
print("max element is {} and the minimun elemnt is {} in book1.csv column 1\n".format(max,min))

#Q2
array_book1.sort()
print(array_book1)

#Q3
array_book1 = array_book1[::-1]
print(array_book1)

#Q4
master_array = list()
master_array.append(array_book1)
array_book2 = np.loadtxt("book2.csv",delimiter= '\t',usecols = 1,skiprows = 1)
master_array.append(array_book2)
array_book3 = np.loadtxt("book2.csv",delimiter= '\t',usecols = 1,skiprows = 1)
master_array.append(array_book3)
print(master_array)
avg = [array_book1.mean(),array_book2.mean(),array_book3.mean()]
print(avg)

#Q5
img = cv2.imread("a.png",1)
print(img)

#Q6
x = cv2.imread("a.png",0)
#cv2.imshow("X",x)
#cv2.waitKey(0)
print(x)

#Q7
y = np.transpose(x)
z = np.matmul(x,y)
print(x)
print(y)
print(z)
#also print time

#Q8
rows = len(x)
cols = len(x[0])
#Using list comprehension to create the array for storing the transpose of x
y1 = [[0 for i in range(rows)] for j in range(cols)]
for i in range(rows):
    for j in range(cols):
        y1[j][i] = x[i][j]
print(y==y1)
"""z= np.array([[0 for i in range(rows)]for j in range(rows)])
for i in range(rows):
    for j in range(rows):
        sum = 0
        for k in range(cols):
            sum = sum + x[i][k]*y[k][j]
        z[i][j]= sum
#also print time"""

#Q9
plt.hist(x,bins=30)
plt.xlabel("Values")
plt.ylabel("Intensity")
plt.title("")
plt.show()

#Q10
start_point= (40,100)
end_point = (70,200)
color = (0,0,0)
thickness = 1
altered_image = cv2.rectangle(x,start_point,end_point,color,thickness)
cv2.imshow("Image after creation of the black box",altered_image)
cv2.waitKey(0)
cv2.destroyAllWindows

#Q11
rows = len(x)
cols = len(x[0])
#array for storing binary values with the threshold 50
Z50 = np.array([[0 for i in range(cols)]for j in range(rows)])
#array for storing binary values with the threshold 70
Z70 = np.array([[0 for i in range(cols)]for j in range(rows)])
#array for storing binary values with the threshold 100
Z100 = np.array([[0 for i in range(cols)]for j in range(rows)])
#array for storing binary values with the threshold 150
Z150 = np.array([[0 for i in range(cols)]for j in range(rows)])
#logical function to store the values into the arrays based on the 
#comparison of the threshold value and initial value stored in array x
for i in range(rows):
    for j in range(cols):
        if x[i][j] >= 50:
            Z50[i][j] = 1
            if x[i][j] >= 70:
                Z70[i][j] = 1
                if x[i][j] >= 100:
                    Z100[i][j] = 1
                    if x[i][j] >= 150:
                        Z150[i][j] = 1
                    else:
                        Z150[i][j] = 0
                else:
                    Z100[i][j] = 0
            else:
                Z70[i][j] = 0
        else:
            Z50[i][j] = 0
print(Z50,Z70,Z100,Z150)
#Q12
#Creating the filter matrix
filter = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
#Using a filter2d function to add the filter to the image
filtered_image = cv2.filter2D(src = x,ddepth=-1,kernel=filter)
#Displaying the filtered image
cv2.imshow("filtered image",filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows
