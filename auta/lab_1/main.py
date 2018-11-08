import cv2
import numpy as np
import pafy
import imutils
from imutils.object_detection import non_max_suppression

url = 'https://www.youtube.com/watch?v=NVay1YJbE0k'

def main():
#     hog = cv2.HOGDescriptor()
#     hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# 
#     im = cv2.imread('pedestrians.jpg')
# 
#     (rects, weights) = hog.detectMultiScale(im, winStride=(4, 4), padding=(8, 8), scale=1.05)
#  
#     for (x, y, w, h) in rects:
#         cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    vPafy = pafy.new(url)
    play = vPafy.getbest(preftype="webm")
    
    cap = cv2.VideoCapture(play.url)
    
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    cnt = 0
    
    while True:
        ret, frame = cap.read()   
        frame = imutils.resize(frame, width=min(400, frame.shape[1]))
        orig = frame.copy() 
        
#         if cnt % 3:
        (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
        
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
 
        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(orig, (xA, yA), (xB, yB), (0, 255, 0), 2)
    
        cv2.imshow('elo', orig)
        cv2.waitKey(1)
        
        cnt += 1

if __name__ == '__main__':
    main()