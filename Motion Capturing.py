import cv2 as cv

video = cv.VideoCapture(0) # for webcam
# video = cv.VideoCapture('/home/chirag/Desktop/Computer Vision/source/lane_video.mp4') # for video file
substractor = cv.createBackgroundSubtractorMOG2(40,30)

while True:
    ret, frame = video.read()
    if ret:
        mask = substractor.apply(frame)
        cv.imshow('Mask', mask)

        if cv.waitKey(5)== ord('1'):
            break
    else:
        # video = cv.VideoCapture('/home/chirag/Desktop/Computer Vision/source/lane_video.mp4')
        video = cv.VideoCapture(0)
        
cv.destroyAllWindows()
video.release()
