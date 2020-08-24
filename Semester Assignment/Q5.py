
import cv2
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
import math


def showBitPlanes(img, num):
        if num != "":
            num = int(num)

        plane1 = img % 2
        plane2 = (img // 2) % 2
        plane3 = (img // 4) % 2
        plane4 = (img // 8) % 2
        plane5 = (img // 16) % 2
        plane6 = (img // 32) % 2
        plane7 = (img // 64) % 2
        plane8 = (img // 128) % 2

        #Resaling between 0 and 1 to convert the image into monocrome

        plane1 = rescale_intensity(plane1, in_range=(0, 255))
        plane1 = (plane1 * 255).astype("uint8")
        plane2 = rescale_intensity(plane2, in_range=(0, 255))
        plane2 = (plane2 * 255).astype("uint8")
        plane3 = rescale_intensity(plane3, in_range=(0, 255))
        plane3 = (plane3 * 255).astype("uint8")
        plane4 = rescale_intensity(plane4, in_range=(0, 255))
        plane4 = (plane4 * 255).astype("uint8")
        plane5 = rescale_intensity(plane5, in_range=(0, 255))
        plane5 = (plane5 * 255).astype("uint8")
        plane6 = rescale_intensity(plane6, in_range=(0, 255))
        plane6 = (plane6 * 255).astype("uint8")
        plane7 = rescale_intensity(plane7, in_range=(0, 255))
        plane7 = (plane7 * 255).astype("uint8")
        plane8 = rescale_intensity(plane8, in_range=(0, 255))
        plane8 = (plane8 * 255).astype("uint8")

        if num == 1:
            cv2.imshow("1 th bit plane", plane1)
        elif num == 2:
            cv2.imshow("2 th bit plane", plane2)
        elif num == 3:
            cv2.imshow("3 th bit plane", plane3)
        elif num == 4:
            cv2.imshow("4 th bit plane", plane4)
        elif num == 5:
            cv2.imshow("5 th bit plane", plane5)
        elif num == 6:
            cv2.imshow("6 th bit plane", plane6)
        elif num == 7:
            cv2.imshow("7 th bit plane", plane7)
        elif num == 8:
            cv2.imshow("8 th bit plane", plane8)
        else:

            fig = plt.figure(figsize=(8, 8))
            columns = 4
            rows = 2

            plt.title("All planes")
            plt.axis('off')
            planes = [plane1, plane2, plane3, plane4, plane5, plane6, plane7, plane8]
            titles = ["first plane", "second plane", "third plane", "fourth plane", "fifth plane", "sixth plane", "seventh plane", "eight plane" ]

            for i in range(1, columns * rows + 1):
                ax = fig.add_subplot(rows, columns, i)
                ax.title.set_text(titles[i-1])
                plt.axis('off')
                plt.imshow(planes[i-1],cmap='gray')
            plt.axis('off')
            plt.show()


def Resconstruct(img, num=0):
    plane1 = img % 2
    plane2 = (img // 2) % 2
    plane3 = (img // 4) % 2
    plane4 = (img // 8) % 2
    plane5 = (img // 16) % 2
    plane6 = (img // 32) % 2
    plane7 = (img // 64) % 2
    plane8 = (img // 128) % 2

    if num == 4:
        img1 = (2 * (2 * (2 * plane8 + plane7) + plane6) + plane5)
        return img1
    elif num == 2:
        img1 = (2 * (2 * (2 * (2 * (2 * (2 * (2 * plane8 + plane7) + 0) + 0) + 0) + 0) + 0) +
                0)
        return img1
    else:
        img1 = (2 * (2 * (2 * (2 * (2 * (2 * (2 * plane8 + plane7) + plane6) + plane5) + plane4) + plane3) + plane2) +
                plane1)
        return img1


def MSE(img1, img2):
    summ = 0
    for i in range (1, img1.shape[0]):
        for j in range(1, img1.shape[1]):
            dif = img1[i, j].astype('float')-img2[i, j].astype('float')
            summ = summ+(dif**2)

    mse = summ/float(img1.shape[0]*img2.shape[1])
    return  mse

def PSNR(mse):
    mul = float((255**2)/mse)
    psnr = float(math.log10(mul))
    return psnr

img = cv2.imread("dollar.tif")
img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
x = input("Enter the number of a specific bit plane to display or Press Enter to Show all: ")
showBitPlanes(img,x)
img1 = Resconstruct(img, 4)
img2 = Resconstruct(img, 2)
cv2.imwrite("reconstructedDollar2.tif", img2)
cv2.imwrite("reconstructedDollar4.tif", img1)

img1 = cv2.imread("reconstructedDollar4.tif")
img1 = cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)

img2 = cv2.imread("reconstructedDollar2.tif")
img2 = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)
rows = 1
cols = 3

images = [img, img1, img2]
fig = plt.figure(figsize=(8, 12))
titles = ["Original", "Reconstructed4Dollar", "Reconstructed2Dollar"]
plt.axis('off')
rows = 3
cols = 1
for i in range(1, rows * cols + 1):

    ax = fig.add_subplot(rows, cols, i)
    ax.title.set_text(titles[i - 1])
    plt.axis('off')
    plt.imshow(images[i - 1], cmap= 'gray')

plt.axis('off')
plt.show()

mse1 = MSE(img, img1)
mse2 = MSE(img, img2)
psnr1 = PSNR(mse1)
psnr2 = PSNR(mse2)
print("Mean Square error of Original Image with reconstructed4Dollar ")
print(mse1)
print("Signal to Noise Ratio of Original Image with reconstructed4Dollar ")
print(psnr1)
print("Mean Square error of Original Image with reconstructed2Dollar")
print(mse2)
print("Mean Square error of Original Image with reconstructed2Dollar")
print(psnr2)
cv2.waitKey(0)





