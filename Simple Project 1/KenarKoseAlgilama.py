
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2

img = cv2.imread('panda.jpg')

cv2.imshow('Original Image', img) 
#cv2.waitKey(0)

#%%
#Resmi Gri dönüştürme
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('Grayscale', gray_image) 
##cv2.waitKey(0)  

#%%
#Resmi siyah beyaza dönüştürme
(thresh, blackAndWhiteImage) = cv2.threshold(gray_image, 127, 225, cv2.THRESH_BINARY)

cv2.imshow('Black white image', blackAndWhiteImage)

#%%
#Bluring (Resmi bulanıklaştırma)
bluring = np.ones((3,3), np.float32)/9
output_bluring = cv2.filter2D(blackAndWhiteImage, -1, bluring)

#output2 = cv2.blur(img, (5, 5))

cv2.imshow('Bluring', output_bluring)

#%%
#Sharping (Keskinleştirme işlemi)
kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharping = cv2.filter2D(output_bluring, -1, kernel_sharpen)

cv2.imshow('Sharping', sharping)


#%%
#Sobel (kenar bulma)
sobel_horizontal = cv2.Sobel(sharping, cv2.CV_64F, 1, 0, ksize=5)
sobel_vertical = cv2.Sobel(sharping, cv2.CV_64F, 0, 1, ksize=5)

cv2.imshow('Sobel', sobel_horizontal)



