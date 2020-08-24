import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image

def my2dGausian(size):
	w, h = size,size
	Mask = [[0] * w for i in range(h)]
	sigma = 10
	sum = 0
	for i in range(w):
		for j in range(h):
			den = 2*(sigma**2)
			neu = ((i-1)**2)+((j-1)**2)
			neu = -neu
			res = neu/den
			res = pow(2.71828, res)
			Mask[i][j] = res
			sum = sum+Mask[i][j]
	Mask = np.array(Mask)
	#Normalizing the Mask
	for i in range(w):
		for j in range(h):
			Mask[i][j] = Mask[i][j] / sum
	return Mask

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
			output[x - pad, y - pad] = k
	return output


image = cv2.imread("pattern.tif")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
kernel9 = my2dGausian(9)
kernel11 = my2dGausian(11)
kernel13 = my2dGausian(13)


#Convolution with NXN filters and converting them into PIL grey scale images

Output9 = Image.fromarray(convolve(image, kernel9))
Output11 = Image.fromarray(convolve(image, kernel11))
Output13 = Image.fromarray(convolve(image, kernel13))
image = Image.fromarray(image)
#ploting code
fig = plt.figure(figsize=(8, 8))
columns = 2
rows = 2

plt.title("Different N X N Gaussian Output")
plt.axis('off')
images = [image, Output9, Output11, Output13]
titles = ["Original", "9 X 9 Gaussian", "11 X 11 Gaussian", "13 X 13 Gaussian"]

for i in range(1, columns * rows + 1):
	ax = fig.add_subplot(rows, columns, i)
	ax.title.set_text(titles[i-1])
	plt.axis('off')
	plt.imshow(images[i-1], cmap='gray')
	plt.axis('off')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()