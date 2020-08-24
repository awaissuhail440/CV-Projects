# import the necessary packages
import numpy as np
import cv2
from PIL import Image

def convolve(image, kernel):
    # convert into grey scale

    kernel = np.flipud(np.fliplr(kernel))  # Flip the kernel

    iR = image.shape[0]
    iC = image.shape[1]
    kR = kernel.shape[0]
    kC = kernel.shape[1]
    #no of pad rows
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

            output[x-pad, y-pad] = k

    return output

gaussian = np.array((
    [1/16, 2/16, 1/16],
    [2/16, 4/16, 2/16],
    [1/16, 2/16, 1/16]), dtype="float")

# load the image in rgb
image = cv2.imread('dollar.tif')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("Applying the Gausian Kernel")
#Getting the Convolution Result
Output = convolve(image, gaussian)

#converting the image and output arrays to PIL images
Output = Image.fromarray(Output)
image = Image.fromarray(image)

#image.show("Original Image")
Output.show("Convolved Image")
cv2.waitKey(0)
cv2.destroyAllWindows()