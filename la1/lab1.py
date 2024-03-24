import matplotlib.pyplot as plt
import numpy as np
from skimage import io

image0 = np.load("images\car_0.npy")
image1 = np.load("images\car_1.npy")
image2 = np.load("images\car_2.npy")
image3 = np.load("images\car_3.npy")
image4 = np.load("images\car_4.npy")
image5 = np.load("images\car_5.npy")
image6 = np.load("images\car_6.npy")
image7 = np.load("images\car_7.npy")
image8 = np.load("images\car_8.npy")


#b
images = [image0, image1, image2, image3, image4, image5, image6, image7, image8]

sumall = 0
for img in images:
    for i in img:
        for j in i:
            sumall += j

print(sumall)

#c

#suma pentru fiecare imagine
sums = []
for img in images:
    sum = 0
    for i in img:
        for j in i:
            sum += j
    sums.append(sum)

#d

max_sum = max(sums)
index = sums.index(max_sum)
print(index)

#e

mean_image = np.mean(images,axis = 0)
io.imshow(mean_image.astype(np.uint8)) 
io.show()

#f

dev_std = np.std(images)
print(dev_std)

#g
for img in images:
    img = np.subtract(img,mean_image)
    img = np.divide(img,dev_std)

for img in images:
    img = img[200:300,280:400]