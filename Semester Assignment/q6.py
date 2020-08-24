import numpy as np,array
import cv2
from PIL import Image
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt

def pixelVal(pix, r1, s1, r2, s2):
    if (0 <= pix and pix <= r1):
        return (s1 / r1) * pix
    elif (r1 < pix and pix <= r2):
        return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2) / (255 - r2)) * (pix - r2) + s2


def contrast_stretched_image(img, r1, r2, s1, s2):
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            type(img[i][j])
            img[i][j] = pixelVal(img[i][j], r1, s1, r2, s2)
    return img


def returnMin(img):
    minn = img[0][0]
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if minn > img[i][j]:
                minn = img[i][j]
    return minn

def returnMax(img):
    maxx = img[0][0]
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if maxx < img[i][j]:
                maxx = img[i][j]
    return maxx

def scale(image):
    rmin = returnMin(image)
    rmax = returnMax(image)
    image = contrast_stretched_image(image, r1=rmin, r2=rmax, s1=0, s2=255)
    return image

def convolve(image, kernel):
    kernel = np.flipud(np.fliplr(kernel))  # Flip the kernel
    iR = image.shape[0]
    iC = image.shape[1]
    kR = kernel.shape[0]
    kC = kernel.shape[1]
    # no of pad rows
    pad = (kC - 1) // 2

    image = np.pad(image, pad, mode='constant', constant_values=0)
    output = np.zeros((iR, iC), dtype="float32")
    iR1 = image.shape[0]
    iC1 = image.shape[1]

    for x in np.arange(pad, iR1 - pad):
        for y in np.arange(pad, iC1 - pad):
            k = 0
            for i in range(0, kR):
                for j in range(0, kC):
                    k = k + (kernel[i][j] * image[x - pad + i, y - pad + j])
            output[x - pad][y - pad] = k
    return output

image = cv2.imread("moon.tif")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
laplacian90 = np.array((
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]), dtype="float")

laplacian45 = np.array((
    [1, 1, 1],
    [1, -8, 1],
    [1, 1, 1]), dtype="float")


img_45 = convolve(image, laplacian45)

img_90 = convolve(image, laplacian90)

img_45s = scale(img_45)
img_90s = scale(img_90)

img_45 = image-0.2*img_45s
img_90 = image-0.2*img_90s

images = [image, img_45s, img_45, image, img_90s, img_90]
titles = ["Original", "Scaled Laplacian", "Sharped Image 45","Original", "Scaled Laplacian", "Sharped Image 90"]

fig = plt.figure(figsize=(8, 8))
columns = 3
rows = 2

plt.axis('off')

for i in range(1, columns * rows + 1):
    ax = fig.add_subplot(rows, columns, i)
    ax.title.set_text(titles[i-1])
    plt.axis('off')
    plt.imshow(images[i-1], cmap='gray')
    plt.axis('off')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()