import numpy as np
import road_detector
import pafy
import cv2

def main():
    vPafy = pafy.new('https://www.youtube.com/watch?v=ZOZOqbK86t0')
    play = vPafy.getbest(preftype="webm")
     
    cap = cv2.VideoCapture(play.url)
    
    while True:
        _, frame = cap.read()
        
        left_lines, right_lines = road_detector.detect(frame)
                
        for line in left_lines:
#             print(line)
            cv2.line(frame, line[0], line[1], (255, 0, 0), 3)
            
            
        for line in right_lines:
#             print(line)
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
        cv2.waitKey(1)
        
#     orig = cv2.resize(orig, (0, 0), fx=0.5, fy=0.5)
    
    
#     print(road_lines)
    
#     cv2.polylines(orig, road_lines, False, (255, 0, 0), 2)
    
#     filtered = cv2.GaussianBlur(orig, (5, 5), 0)
#     cv2.imshow('filtered', filtered)
#     
#     edges = cv2.Canny(filtered, 100, 200)    
#     cv2.imshow('edges', edges)
# 
#     lines = cv2.HoughLines(edges,1,np.pi/180,200)
#     horizont_line = None
#     
#     #Find horizont line
#     for line in lines:
#         for rho,theta in line:
#             theta_deg = theta * 180 / np.pi
#             
#             if theta_deg < 91 and theta_deg > 89:
#                 horizont_line = line
#                 break
#     
#     #Find road borders
#     road_borders = {'left': None, 'right': None}
#     
#     for line in lines:
#         for rho,theta in line:
#             theta_deg = theta * 180 / np.pi
#             
#             if theta_deg > 60 and theta_deg < 70:
#                 if road_borders['left'] is None:
#                     road_borders['left'] = line    
#             elif theta_deg > 110 and theta_deg < 120:
#                 if road_borders['right'] is None:
#                     road_borders['right'] = line
#             
#             if road_borders['left'] is not None and \
#                road_borders['right'] is not None:
#                 break
#           
#     #Find crossing point
#     road_eq = []
#     for line in road_borders.values():
#         if line is not None:
#             for rho,theta in line:
#                 a = np.cos(theta)
#                 b = np.sin(theta)
#                 x0 = a*rho
#                 y0 = b*rho
#                 x1 = int(x0 + 1000*(-b))
#                 y1 = int(y0 + 1000*(a))
#                 x2 = int(x0 - 1000*(-b))
#                 y2 = int(y0 - 1000*(a))
#                 
#                 print((x1,y1), (x2,y2))
#                 
#                 cv2.line(orig,(x1,y1),(x2,y2),(0,0,255),2)
#                 
# #                 y1 = orig.shape[0] - y1
# #                 y2 = orig.shape[0] - y2
#                 
#                 A1 = (y1 - y2) / (x1 - x2)
#                 B1 = (y1 - (y1 - y2)/(x1 - x2)*x1)
#                 
#                 road_eq.append((A1, B1))
#                
#     A1 = road_eq[0][0]
#     B1 = -1
#     C1 = road_eq[0][1]
#     A2 = road_eq[1][0] 
#     B2 = -1
#     C2 = road_eq[1][1]
#     
#     print(road_eq)
#     
#     print(A1, B1, C1)
#     print(A2, B2, C2)
#     
#     W = A1*B2 - A2*B1
#     Wx = (-C1)*B2 - (-C2)*B1
#     Wy = A1*(-C2) - A2*(-C1)
#     
#     X = int(Wx/W)
#     Y = int(Wy/W)
#     
#     print(X, Y)
#     
#     cv2.circle(orig, (X, Y), 3, (255, 0, 0))
                
#     cv2.imshow('orig', orig)
#     cv2.waitKey()

if __name__ == '__main__':
    main()