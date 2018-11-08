import numpy as np
import cv2

def _get_im_strip(im, strip_no, total_strips):
    im_height = im.shape[0]
    strip_height = im_height / total_strips
    start_height = int(im_height - strip_no * strip_height)
    end_height = int(start_height - strip_height)
    return im[end_height:start_height]

def _hough_line_to_slope_form(rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 100*(-b))
    y1 = int(y0 + 100*(a))
    x2 = int(x0 - 100*(-b))
    y2 = int(y0 - 100*(a))
    
    if x1 == x2:
        A = 0
        B = x1
    else:
        A = (y2 - y1) / (x2 - x1)
        B = y1 - (y2 - y1)/(x2 - x1)*x1
        
    if A == 0:
        A = 0.001

    return (A, B)

def detect(im):
    detected_lines = []
    
    max_strips = 7
    
    roi = im[int(im.shape[0] * 0.55):]
    filtered = cv2.GaussianBlur(roi, (5, 5), 0)
    edges = cv2.Canny(filtered, 100, 200)
    cv2.imshow('canny', edges)
    
    for i in range(max_strips):
        strip = _get_im_strip(edges, i, max_strips)
        hough_lines = cv2.HoughLines(strip,1,np.pi/180,20)
         
        if hough_lines is not None:    
            for line in hough_lines:
                for rho,theta in line:
                    theta_deg = theta * 180 / np.pi
                    
                    if theta_deg < 70 or theta_deg > 110:
                        (A, B) = _hough_line_to_slope_form(rho, theta)
                        
                        #Translate B
                        y_translation = im.shape[0] - (i + 1) * strip.shape[0]
                        B += y_translation
        
                        y_start = y_translation
                        x_start = (y_start - B) / A
                        
                        y_end = y_start + strip.shape[0]
                        x_end = (y_end - B) / A
                       
                        start = (int(x_start), int(y_start))
                        end = (int(x_end), int(y_end))
                        
                        detected_lines.append((start, end))
                        
    return detected_lines
    
#     for i in range(max_strips):
#         new_left_line = None
#         new_right_line = None
#         
#         strip = _get_im_strip(im, i, max_strips)
#         
#         filtered = cv2.GaussianBlur(im, (5, 5), 0)
# #         cv2.imshow('filtered', filtered)
#          
#         edges = cv2.Canny(filtered, 100, 200)    
# #         cv2.imshow('edges', edges)
#      
#         lines = cv2.HoughLines(edges,1,np.pi/180,200)
#         
#         if lines is not None:    
#             for line in lines:
#                 for rho,theta in line:
#                     theta_deg = theta * 180 / np.pi
#                     a = np.cos(theta)
#                     b = np.sin(theta)
#                     x0 = a*rho
#                     y0 = b*rho
#                     x1 = int(x0 + 1000*(-b))
#                     y1 = int(y0 + 1000*(a) + im.shape[0] - strip.shape[0] * i)
#                     x2 = int(x0 - 1000*(-b))
#                     y2 = int(y0 - 1000*(a) + im.shape[0] - strip.shape[0] * i)
#                       
#                     if theta_deg > 60 and theta_deg < 70:
#                         if new_left_line is None:
#                             new_left_line = [(x1, y1), (x2, y2)]
#                             print(new_left_line)
#                     elif theta_deg > 110 and theta_deg < 120:
#                         if new_right_line is None:
#                             new_right_line = [(x1, y1), (x2, y2)]
#                             print(new_right_line)
#                       
#                     if new_left_line is not None and \
#                        new_right_line is not None:
#                         left_line += new_left_line
#                         right_line += new_right_line
#                         break
#     
#     return (np.array(left_line), np.array(right_line))