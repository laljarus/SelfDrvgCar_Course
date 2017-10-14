# Finding Lane Lines on the Road

## Reflection

## Description

The aim of this project is to find the lane markings on the road. The program gets the raw video of the road as input and processes every frame in the video and return a annotated video where the lanes marking are highligted. The pipelines uses the canny edge detection algorithm to detect the edges and uses hough transform to find the location of the lines in the edges of the image. The pipeline has the steps expained as follows

[//]: # (Image References)

[image1]: ./test_images/whiteCarLaneSwitch.jpg "raw_image"
[image2]: ./test_images_output/grayscale.jpg
[image3]: ./test_images_output/edges.jpg
[image4]: ./test_images_output/masked_edges.jpg
[image5]: ./test_images_output/lane_markings.jpg
[image6]: ./test_images_output/lane_markings_smooth.jpg

1. Image is converted to gray scale inorder to be used in canny edge detection algorithm.

### Grayscale
![grayscale][image2]

2. The image is filtered using Gaussian filter and then Canny edge detection algorithm is used to find the edges in the image.
### Edges
![canny edge][image3]

3. The image is then masked to show only region of interest where the lanes could be prasent. For performing this the "polyfill" function is used. The a trapezoidal region in the bottom half of the image is selected as this is the region where road is found.
### Masked Edges
![masked edge][image4]

4. The hough tranform is performed in the masked edges image. Hough transform is used to find the line segments present in the masked edge image. Then these line segments are drwan on the orginal image. The line segments found by hough transform are the lane markings but there could also be some lines other than the lane markings. So these lines have to be averaged and extrapolated which is done in the next step.

### Lane Markings
![lane_markings][image5]

5. As final step the lane marking found using hough transform is smoothened and extrapolated. In order to do this the lines found with hough transform are split into two groups based on the slope of the lines. In reality the lanes in the road are parallel to each other but in two dimensional image the parallel lanes appear to meet at far away distance. This is because the sizes of objects appear to be smaller when they are far away. Because of this effect the lines to the left of camera has opposite polarity when compared to the lines on the right of the camera. Based on the polarity the lines are split in two groups where one group contains lines on the left side and the other group contains lines on the right side. The line segments grouped may also contain other noises such as road edges and edges of the object on the road and these have to be filtered out. To do this mean of slopes of both the group of line segments are calculated. It is then compared with each line segment and if the relative difference between the slopes is beyond a threshold the line segment is removed from the group as it could be the edge of other objects in the road. This way the line segments are filtered and only the line segments of the lanes remain. Then finally the a new averaged position of the line segements are calculated and two lines are drown on the orginal image which are the estimate of lane markings.

### Smoothened Lane Markings

![smooth lane marking][image6]

## Shortcomings

In each step of the pipelines there are parameters used and these paramters are tuned using trial and error method for the given set of the images and videos. The given set of images were taken during normal weather and light conditions. In case the pipeline is used for different weather and light conditions it pipeline may not work as it is sensitive to the parameters used. 

The implemented pipeline did not estimate the position of lanes for the video names "challenge". The test video "challenge" has many objects close to the camera such cars in the next lane, trees on the sides, road patches and extension of the own car etc. The pipeline cannot filter out these non relevent objects.

## Possible Improvements

In order to use the algorithm for different light and weather conditions the algorithm can be extended to use different sets of parameters for different conditions.

The algorithm can be extented to run multiple iterations be narrowing the masking region based on the solution of the previous iteration. Using multiple iteration can filter out non relavant objects as the masking region gets smaller.









