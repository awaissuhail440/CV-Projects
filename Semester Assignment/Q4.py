
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
total_pixels = 0
grey_level = 255

def equilization(image, histogram, total_pixels, Array):
    #probability
    new_arr = [x/total_pixels for x in histogram]
    #commulative sum
    for i in range(1,256):
        new_arr[i] = new_arr[i]+new_arr[i-1]
    #multiplication with grey levels
    new_arr = [x * 255 for x in new_arr]
    new_arr = [round(x) for x in new_arr]

    for i in range (0, 256):
        for (a, b) in Array[i]:
            image[a][b] = new_arr[image[a][b]]
    return image

def applyHistogramEquilization(image):
    histogram = [0] * 256
    A = np.array([] * 255)
    iR = image.shape[0]
    iC = image.shape[1]

    # getting number of pixels of intensities
    total_pixels = 0
    Array = []
    for i in range(0, 256):
        subarray = []
        Array.append(subarray)

    for i in range(0, iR):
        for j in range(0, iC):
            total_pixels = total_pixels + 1
            histogram[image[i][j]] = histogram[image[i][j]] + 1
            x = i
            y = j
            Array[image[i][j]].append((i, j))
    image = equilization(image, histogram, total_pixels, Array)
    return image

def pixelVal(pix, r1, s1, r2, s2):
    if (0 <= pix and pix <= r1):
        return (s1 / r1) * pix
    elif (r1 < pix and pix <= r2):
        return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2) / (255 - r2)) * (pix - r2) + s2


def contrast_stretched_image(image, r1, r2, s1, s2):
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            image[i][j] = pixelVal(image[i][j], r1, s1, r2, s2)
    return image


def returnMin(image):
    minn = image[0][0]
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if minn > image[i][j]:
                minn = image[i][j]
    return minn

def returnMax(image):
    maxx = image[0][0]
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if maxx < image[i][j]:
                maxx = image[i][j]
    return maxx


imge = cv2.imread('pollen.tif')
imge = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY)

r1 = 100
s1 = 50
r2 = 200
s2 = 230


print("Contrast Stretching where (r1,s1) = (100,50) and (r2,s2) = (200,230)")
img1 = imge.copy()
img1 = contrast_stretched_image(img1, r1, r2, s1, s2)

s1 = 0
s2 = 255

print("Contrast Stretching where (r1,s1) = (rmin,0) and (r2,s2) = (rmax,255)")

r1 = returnMin(imge)
r2 = returnMax(imge)
img2 = imge.copy()
img2 = contrast_stretched_image(img2, r1, r2, s1, s2)

print("Applying Histogram Equilization ")
img3 = imge.copy()
img3 = applyHistogramEquilization(img3)

images = [imge, img1, img2, img3]
titles = ["Original", "Contrast Stretched", "Contrast Stretched(min,max)", "Histogram Equilization"]
rows = 1
columns = 4
fig = plt.figure(figsize=(20, 20))
for i in range(1, columns * rows + 1):
    ax = fig.add_subplot(rows, columns, i)
    ax.title.set_text(titles[i-1])
    plt.axis('off')
    plt.imshow(images[i-1], cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
