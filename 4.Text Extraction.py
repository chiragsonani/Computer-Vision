"""
Page segmentation modes:

0.  Orientation and script detection (OSD) only.
# 1.  Automatic page segmentation with OSD.
2.  Automatic page segmentation, but no OSD, or OCR.
3.  Fully automatic page segmentation, but no OSD. (Default)
4.  Assume a single column of text of variable sizes.
5.  Assume a single uniform block of vertically aligned text.
6.  Assume a single uniform block of text.
7.  Treat the image as a single text line.
8.  Treat the image as a single word.
9.  Treat the image as a single word in a circle.
10. Treat the image as a single character.
11. Sparse text. Find as much text as possible in no particular order.
12. Sparse text with OSD.
13. Raw Line. Treat the image as a single text Line, bypassing hacks that are Tesseract-specific.

OCR Engine modes:
0.  Legacy engine only.
1.  Neural nets LSTM engine only.
2.  Legacy + LSTM engines.
3.  Default, based on what is available. 
"""
import pytesseract
from PIL import Image
from pytesseract import Output
import cv2

# text = pytesseract.image_to_string(Image.open('/home/chirag/Desktop/Computer Vision/source/Text.png'), config=myconfig)
# print(text)

img = cv2.imread('/home/chirag/Desktop/Computer Vision/source/Logos.png')

img1 = img.copy()
myconfig1 = r'--oem 3 --psm 6'
height, width, _ = img1.shape

boxes = pytesseract.image_to_boxes(img, config=myconfig1)
for b in boxes.splitlines():
    b = b.split(' ')
    img1 = cv2.rectangle(img1, (int(b[1]), height - int(b[2])), (int(b[3]), height - int(b[4])), (0,255,0), 2) #boxes for individual characters

cv2.imshow('img1', img1)

img2 = img.copy()
myconfig2= r'--oem 3 --psm 11'
data = pytesseract.image_to_data(img2, config=myconfig2, output_type=Output.DICT)
# print(data.keys())
# print(data['text'])
total_boxes = len(data['text'])

for i in range(total_boxes):
    if float(data['conf'][i]) > 50:   #draw box only for certain confidence
        (x, y, width, height) = (data ['left'][i], data['top'][i], data['width'][i], data['height'][i])
        img2 = cv2.rectangle(img2, (x, y), (x+width, y+height), (0, 255, 0), 2)                                   #boxes around each word
cv2.imshow('img2',img2)

cv2.waitKey(0)
