#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#%matplotlib inline

#import image
image = mpimg.imread('test_images/solidYellowCurve2.jpg')
#printing out some stats and plotting

print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')# -*- coding: utf-8 -*-


# Definition of helper functions

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img, lines

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def slope_intercept(lines,imshape):
    
    lines = lines[:,0,:]    
    lines_left = []
    lines_right = []
        
    
    for x1,y1,x2,y2 in lines:
        slope = (y2-y1)/(x2-x1)
        x_intercept = y1-slope*x1        
        if slope < 0:  
            lines_left.append([slope,x_intercept])            
        else:
            lines_right.append([slope,x_intercept]) 
            
    lines_left = np.asarray(lines_left)
    lines_right = np.asarray(lines_right)       
    
        
    return lines_left,lines_right

"""
def filter_lines(dict_lines): 
    
    mean_slopes = sum(dict_lines)/len(dict_lines)    
    
    for line_slope in dict_lines:
            diff = (mean_slopes - line_slope)/mean_slopes            
            if diff > 0.1:
                dict_lines.pop(line_slope)
"""               
    
            

# Processing the image
    
# covert the color image to gray scale
gray = grayscale(image)

# Apply gaussian blur to the gray image
kernel_size = 5
blur_gray = gaussian_blur(image, kernel_size)

# Canny edge detection
low_threshold = 50
high_threshold = 150
edges = canny(image,low_threshold,high_threshold)

# Masking relevent region

imshape = image.shape
width = imshape[1]
height = imshape[0]
quad_parm = 0.11

#height_quad = 0.1*height
#width_quad = height_quad*height/width


vertices = np.array([[(0,height),(width*(0.5-quad_parm),height*(0.5+quad_parm)),(width*(0.5+quad_parm),height*(0.5+quad_parm)),(width,height)]], dtype=np.int32)

masked_edges = region_of_interest(edges,vertices)

rho = 2 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 15     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 40 #minimum number of pixels making up a line
max_line_gap = 20    # maximum gap in pixels between connectable line segments

[line_img,lines] = hough_lines(masked_edges,rho,theta,threshold,min_line_length,max_line_gap)

[lines_left,lines_right] = slope_intercept(lines,imshape)

lane_left = np.mean(lines_left,axis=0)
lane_right = np.mean(lines_right,axis=0)
    
y1 = imshape[0]
y2 = imshape[0]*0.6  
    
x1_left = (y1-lane_left[1])/lane_left[0]
x2_left = (y2-lane_left[1])/lane_left[0]
    
x1_right = (y1-lane_right[1])/lane_right[0]
x2_right = (y2-lane_right[1])/lane_right[0]
    
lanes = np.array(([[[x1_left,y1,x2_left,y2]],[[x1_right,y1,x2_right,y2]]]),dtype=np.int32)
#filtered_left_lines = filter_lines(lines_left)

avg_line_img = np.zeros((imshape[0], imshape[1], 3), dtype=np.uint8)
draw_lines(avg_line_img, lanes)

raw_line_edges = weighted_img(avg_line_img,image,0.8,1,0)

plt.imshow(raw_line_edges)