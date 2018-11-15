import numpy as np
import road_detector
import cv2


def main():
    frame = cv2.imread('frames/frame_600.png')
    
    left_lines, right_lines = road_detector.detect(frame)
            
    for line in left_lines:
        cv2.line(frame, line[0], line[1], (255, 0, 0), 3)
        
        
    for line in right_lines:
        cv2.line(frame, line[0], line[1], (0, 0, 255), 3)
    
    road_poly = []
    
    for line in right_lines:
        road_poly.append(line[1])
        road_poly.append(line[0])
        
    for line in reversed(left_lines):
        road_poly.append(line[0])
        road_poly.append(line[1])
    
    if len(road_poly) > 0:
        cv2.fillConvexPoly(frame, np.asarray(road_poly), (0, 255, 0))
                       
    cv2.imshow('frame', frame)
    
    
    #Draw detected road on black background
    road = np.zeros(frame.shape[:-1], np.uint8)
    cv2.fillConvexPoly(road, np.asarray(road_poly), 255)
    cv2.imshow('road', road)
    
    gt = cv2.imread('frames/frame_600_GT.png')
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gt', gt)
    
    all_255 = np.ones(frame.shape[:-1], np.uint8) * 255
    all_0 = np.zeros(frame.shape[:-1], np.uint8)
    
    tp = np.count_nonzero(np.logical_and(road == gt, road == all_255))
    tn = np.count_nonzero(np.logical_and(road == gt, road == all_0))
    fp = np.count_nonzero(np.logical_and(road != gt, road == all_255))
    fn = np.count_nonzero(np.logical_and(road != gt, road == all_0))

    print('tp={}, tn={}, fp={}, fn={}'.format(tp, tn, fp, fn))
 
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (prec * recall) / (prec + recall)
    
    print('prec={}, recall={}, f1={}'.format(prec, recall, f1))
       
    cv2.waitKey()
        
if __name__ == '__main__':
    main()