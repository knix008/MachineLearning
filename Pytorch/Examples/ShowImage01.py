import cv2
import matplotlib.pyplot as plt

img = cv2.imread('wallpaper.jpg')
img = img[50:250,40:240,:]
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

try:
    plt.show(img)
except:
    print("Cannot show the image!!!")
    
print(img.shape)
# (200,200,3)