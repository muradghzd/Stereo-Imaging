################################################################
# WARNING
# --------------------------------------------------------------
# When you submit your code, do NOT include blocking functions
# in this file, such as visualization functions (e.g., plt.show, cv2.imshow).
# You can use such visualization functions when you are working,
# but make sure it is commented or removed in your final submission.
#
# Before final submission, you can check your result by
# set "VISUALIZE = True" in "hw3_main.py" to check your results.
################################################################
from locale import normalize
from turtle import shape
from utils import normalize_points
import numpy as np
import cv2


#=======================================================================================
# Your best hyperparameter findings here
WINDOW_SIZE = 7
DISPARITY_RANGE = 40
AGG_FILTER_SIZE = 5



#=======================================================================================
def bayer_to_rgb_bilinear(bayer_img):
    ################################################################
    # Initialize channels and rgb_img
    m, n = bayer_img.shape
    rgb_img = np.zeros((m,n,3))
    r_img = np.zeros((m,n))
    g_img = np.zeros((m,n))
    b_img = np.zeros((m,n))
    
    # Extract channels from the raw image
    r_img[0:m:2, 0:n:2] = bayer_img[0:m:2, 0:n:2]
    g_img[0:m:2, 1:n:2] = bayer_img[0:m:2, 1:n:2]
    g_img[1:m:2, 0:n:2] = bayer_img[1:m:2, 0:n:2]
    b_img[1:m:2, 1:n:2] = bayer_img[1:m:2, 1:n:2]

    #Define kernels
    kernel1 = np.array([[0,1,0],[0,0,0],[0,1,0]])/2
    kernel2 = np.array([[0,0,0],[1,0,1],[0,0,0]])/2
    kernel3 = np.array([[0,1,0],[1,0,1],[0,1,0]])/4

    # Interpolating red channel
    r_img = r_img + cv2.filter2D(r_img, -1, kernel1)
    r_img = r_img + cv2.filter2D(r_img, -1, kernel2)

    # Interpolating green channel
    g_img = g_img + cv2.filter2D(g_img, -1, kernel3) 

    # Interpolating blue channel
    b_img = b_img + cv2.filter2D(b_img, -1, kernel1)
    b_img = b_img + cv2.filter2D(b_img, -1, kernel2)

    rgb_img[:,:,0] = r_img
    rgb_img[:,:,1] = g_img
    rgb_img[:,:,2] = b_img    
    ################################################################
    return rgb_img



#=======================================================================================
def bayer_to_rgb_bicubic(bayer_img):
    # Your code here
    ################################################################
    rgb_img = None


    ################################################################
    return rgb_img



#=======================================================================================
def calculate_fundamental_matrix(pts1, pts2):
    # Assume input matching feature points have 2D coordinate
    assert pts1.shape[1]==2 and pts2.shape[1]==2
    # Number of matching feature points should be same
    assert pts1.shape[0]==pts2.shape[0]
    # Your code here
    ################################################
    pts1, T1 = normalize_points(pts1.T, 2)
    pts2, T2 = normalize_points(pts2.T, 2)

    pts1_x = pts1[0] 
    pts1_y = pts1[1]
    pts2_x = pts2[0]
    pts2_y = pts2[1]
    
    A = np.concatenate((pts2_x*pts1_x, pts2_x*pts1_y, pts2_x,
                        pts2_y*pts1_x, pts2_y*pts1_y, pts2_y,
                        pts1_x, pts1_y, np.ones(8)))
    A = np.reshape(A, (9,8))
    AT_A = A @ A.T
    eig_val, eig_vec = np.linalg.eig(AT_A)
    min_idx = np.argsort(eig_val)[0]
    f = eig_vec[:, min_idx]
    F = np.reshape(f, (3,3))   
    S, V, D = np.linalg.svd(F)
    V[2] = 0
    fundamental_matrix = S @ np.diag(V) @ D
    fundamental_matrix = T2.T @ fundamental_matrix @ T1
    ################################################################
    return fundamental_matrix



#=======================================================================================
def rectify_stereo_images(img1, img2, h1, h2):
    # Your code here
    # You should get un-cropped image.
    # In order to superpose two rectified images, you need to create certain amount of margin.
    # Which means you need to do some additional things to get fully warped image (not cropped).
    ################################################
    img1_rectified = None
    img2_rectified = None

    ################################################
    return img1_rectified, img2_rectified




#=======================================================================================
def calculate_disparity_map(img1, img2):
    # First convert color image to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    # You have to get disparity (depth) of img1 (left)
    # i.e., I1(u) = I2(u + d(u)),
    # where u is pixel positions (x,y) in each images and d is dispairty map.
    # Your code here
    ################################################
    disparity_map = None
    


    ################################################################
    return disparity_map


#=======================================================================================
# Anything else:
