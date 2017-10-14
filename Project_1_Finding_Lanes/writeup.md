# Finding Lane Lines on the Road

## Reflection

The aim of this project is to find the lane markings on the road. The program gets the raw video of the road as input and processes every frame in the video and return a annotated video where the lanes marking are highligted. The pipelines uses the canny edge detection algorithm to detect the edges and uses hough transform to find the location of the lines in the edges of the image. The pipeline has the steps expained as follows

[//]: # (Image References)

[image1]: ./test_images/solidWhiteCurve.jpg "Raw Image"


1. Image is converted to gray scale inorder to be used in canny edge detection algorithm.


![alt text][image1]


