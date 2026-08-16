import cv2
import scipy.sparse as sp
import numpy as np


def edge_det(type, img): 
    if type == "sobel":
        # Apply Sobel Algorithm
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3) 
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3) 
        gradient_magnitude = cv2.magnitude(sobelx, sobely)
        gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
        return gradient_magnitude
    
    if type == "canny": 
        # Apply Canny Algorithm
        blur = cv2.GaussianBlur(img, (5, 5), 1.4)
        edges = cv2.Canny(blur, threshold1=100, threshold2=200)
        cv2.imshow("Canny Edge Detection", edges)
        cv2.imwrite("../data/unit_test/canny.png", edges)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
