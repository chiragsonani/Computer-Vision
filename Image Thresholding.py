import cv2 as cv
# import numpy as np

img = cv.imread('/home/chirag/Desktop/Computer Vision/source/book_page.png')
img = cv.cvtColor (img, cv.COLOR_BGR2GRAY)

_, result = cv.threshold(img, 110, 255, cv.THRESH_BINARY) #Binary Thresholding/Static Threshold (B/W pixels)

adaptive = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 51, 1) #Adaptive Thresholding

cv.imshow('Original', img)
cv.imshow('Static Threshold', result)
cv.imshow('Adaptive Threshold', adaptive)

cv.waitKey(0)
cv.destroyAllWindows()