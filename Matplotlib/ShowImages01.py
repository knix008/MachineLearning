import matplotlib.pyplot as plt
import matplotlib.image as img

fig = plt.figure()
rows = 1
cols = 2

img1 = img.imread('Sample01.jpg')
img2 = img.imread('Sample02.jpg')
 
ax1 = fig.add_subplot(rows, cols, 1)
ax1.imshow(img1)
ax1.set_title('Jumok community')
ax1.axis("off")
 
ax2 = fig.add_subplot(rows, cols, 2)
ax2.imshow(img2)
ax2.set_title('Withered trees')
ax2.axis("off")
 
plt.show()