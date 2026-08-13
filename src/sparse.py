import cv2
import scipy.sparse as sp
import numpy as np


def edge_det(type): 
    img = cv2.imread('../data/unit_test/img1.jpg', cv2.IMREAD_GRAYSCALE)

    if type == "sobel":
        # Apply Sobel Algorithm
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges
        gradient_magnitude = cv2.magnitude(sobelx, sobely)
        gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
        cv2.imshow("Sobel Edge Detection", gradient_magnitude)
        cv2.imwrite("../data/unit_test/sobel.png", gradient_magnitude)

    if type == "canny": 
        # Apply Canny Algorithm
        blur = cv2.GaussianBlur(img, (5, 5), 1.4)
        edges = cv2.Canny(blur, threshold1=100, threshold2=200)
        cv2.imshow("Canny Edge Detection", edges)
        cv2.imwrite("../data/unit_test/canny.png", edges)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sparse(): 
    img = cv2.imread('../data/mask_test/canny.png', cv2.IMREAD_GRAYSCALE)
    if img is None: 
        raise ValueError("Cant find image")
    sparse_matrix = sp.csr_matrix(img)
    sp.save_npz('../data/sparse_test/canny_sparse.npz', sparse_matrix)

sparse()