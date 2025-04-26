import cv2 as cv
import numpy as np

camera = cv.VideoCapture(0)

while True:
    ret, frame = camera.read()
    cv.imshow('Camera', frame)                 # Display the feed (original frame)

    laplacian = cv.Laplacian(frame, cv.CV_64F) # Laplacian filter applied on frame
    laplacian = np.uint8(laplacian)            # Convert float output in the int range of 0-255
    cv.imshow('Laplacian', laplacian)          # Display the Laplacian output

    edges = cv.Canny(frame, 30, 30)          # Canny edge detection
    cv.imshow('Canny', edges)                  # Display the Canny output

    if cv.waitKey(5) == ord('1'):
        break

camera.release()
cv.destroyAllWindows()

