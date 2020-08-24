#libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

#reading the image
image = cv2.imread("sudoku.png")

#for drawing lines
original = image

#convert image into gray scale
image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

#showing the image
cv2.imshow("Origional Image",original)

#input from user for n max lines
max = input("Enter the n for n max lines..Enter 0 to display lines greater than and equal to threshold of 200 pixels: ")
max = int(max) #convert into int

#find the edge map of the image

#getting rows and cols of image
row = image.shape[0]
col = image.shape[1]

#edges with HighThreshold(HT) = 250 and LowThreshold(LT) = 200
edges = cv2.Canny(image,50,200)
cv2.imshow("Edge Map",edges)

#Create an Accumulator Array

#range of angles in radians from range -90 degrees to 90 degrees
thetas = np.deg2rad(np.arange(-90.0, 91.0))

#Range of rho's in acculmulator array that is the 2D D is the diagonal distance

#using distance formula
D = np.sqrt((row-1-0)*(row-1-0)+(col-1-0)*(col-1-0)) #x1,y1 = 0,0 #x2,y2 = row,col

#D can be a floating point number so we map it to the nearest integer
D = int(D)

#range of rhos
rhos = np.array(np.arange(-D,D+1))

#accumulator array rows and colums;
acc_rows,acc_cols = len(rhos),len(thetas)

#initialize the accumulator array with zeros
accumulator_array = np.zeros((acc_rows,acc_cols), dtype=np.uint64)

#array of points
points = []
for i in range(0,acc_rows):
    points.append([])
    for j in range(0,acc_cols):
        points[i].append([])


#fill the acculmulator arrays with votes for every edge point

#arrays of sines and cosines
cosines = np.cos(thetas)
sines = np.sin(thetas)


#for voting
for x in range(0,row):
    for y in range(0,col):
        #for every edge point
        if edges[x][y] == 255:
            #for every theta in range of thetas
            for i in range(0,len(thetas)):
                #calculate rho
                    rho = int(x*cosines[i]+y*sines[i])+D
                    accumulator_array[rho][i] = accumulator_array[rho][i]+1
                    points[rho][i].append((y,x))


cv2.imwrite("accumulator.png",accumulator_array)
accumulator = cv2.imread("accumulator.png",0)

#displaying the lines greater than specific threshold
if(max == 0):
    for i in range(0,acc_rows):
        for j in range(0,acc_cols):
            if(accumulator_array[i][j] >= 200):
                points_arr = points[i][j]
                p1 = points_arr[0]
                p2 = points_arr[len(points_arr)-1]
                original = cv2.line(original,p1,p2,(0,255,0),1)
else:
    indices = accumulator.argpartition(accumulator_array.size - max, axis=None)[-max:]
    x, y = np.unravel_index(indices, accumulator_array.shape)
    for i, j in zip(x,y):
        points_arr = points[i][j]
        p1 = points_arr[0]
        p2 = points_arr[len(points_arr) - 1]
        original = cv2.line(original, p1, p2, (0, 255, 0), 1)

cv2.imshow("Detected Lines",original)

fig, ax = plt.subplots(constrained_layout=True)
plt.title("Original Accumulator Array")
plt.imshow(accumulator,cmap="gray"),
plt.axis('off'),
plt.show()

# resize image
scale_percent = 750 # percent of original size
width = int(accumulator.shape[1] * scale_percent / 100)
height = int(accumulator.shape[0])
dim = (width, height)
resized = cv2.resize(accumulator, dim, interpolation = cv2.INTER_AREA)
fig, ax = plt.subplots(constrained_layout=True)
plt.title("Accumulator Array Resized")
plt.imshow(resized,cmap="gray"),
plt.axis('off'),
plt.show()

cv2.waitKey(0)
