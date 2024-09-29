import cv2
import matplotlib.pyplot as plt

fig = plt.figure()
rows = 1
cols = 2

img1 = cv2.imread('Sample01.jpg')
img2 = cv2.imread('Sample02.jpg')

#gif = cv2.VideoCapture('Sample01.gif')
#img2 = cv2.imread('Sample02.png')

#ret, frame = gif.read()
#if ret: 
#   print("Read GIF OK!!!")
#else:
#    print("Cannot read GIF!!!")
#    exit(1)
    
#img1 = frame

ax1 = fig.add_subplot(rows, cols, 1)
ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
ax1.set_title('Jumok community')
#ax1.axis("off")
 
ax2 = fig.add_subplot(rows, cols, 2)
ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
ax2.set_title('Withered trees')
#ax2.axis("off")
 
plt.show()