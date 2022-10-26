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
    rgb_img = None
    # print(f"Matrix: {bayer_img}")
    # print(f"Shape of bayer img: {bayer_img.shape}")
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
    x1 = np.array([*pts1[2], 1])
    x2 = np.array([*pts2[2], 1])
    x1 = np.reshape(x1, (1,3))
    x2 = np.reshape(x2, (3,1))
    
    pts1, T1 = normalize_points(pts1.T, 2)
    pts2, T2 = normalize_points(pts2.T, 2)
    print(f"Shape of T1: {T1.shape}")
    pts1_x = pts1[0] 
    pts1_y = pts1[1]
    pts2_x = pts2[0]
    pts2_y = pts2[1]
    
    A = np.concatenate((pts2_x*pts1_x, pts2_x*pts1_y, pts2_x,
                        pts2_y*pts1_x, pts2_y*pts1_y, pts2_y,
                        pts1_x, pts1_y, np.ones(8)))
    print(f"Shape of A: {A.shape}")
    A = np.reshape(A, (8,9))
    AT_A = A.T @ A
    eig_val, eig_vec = np.linalg.eig(AT_A)
    min_idx = np.argsort(eig_val)[0]
    f = eig_vec[:, min_idx]
    F = np.reshape(f, (3,3))   
    S, V, D = np.linalg.svd(F)
    print(f"V vales: {V}")
    V[2] = 0
    # fundamental_matrix = np.matmul(np.matmul(S, np.diag([*V[:2], 0])), D)
    fundamental_matrix = S @ np.diag(V) @ D
    fundamental_matrix = T2.T @ fundamental_matrix @ T1
    
    print(f"shape of x1: {x1.shape}") 
    print(f"shape of x2: {x2.shape}")
    print(f"sihape of fundamental matrix: {fundamental_matrix.shape}")
    res = x1 @ fundamental_matrix @ x2
    # res = x1@fundamental_matrix@x2
    # res = np.matmul(x1, fundamental_matrix, x2)
    print(f"Result: {res}")
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
