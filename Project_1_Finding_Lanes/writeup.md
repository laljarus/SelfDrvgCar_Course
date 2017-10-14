# Finding Lane Lines on the Road

## Reflection

The aim of this project is to find the lane markings on the road. The program gets the raw video of the road as input and processes every frame in the video and return a annotated video where the lanes marking are highligted. The pipelines uses the canny edge detection algorithm to detect the edges and uses hough transform to find the location of the lines in the edges of the image. The pipeline has the steps expained as follows

[//]: # (Image References)

[image1]: ./test_images/whiteCarLaneSwitch.jpg "raw_image"
[image2]: ./test_images_output/grayscale.jpg
[image3]: ./test_images_output/edges.jpg
[image4]: ./test_images_output/masked_edges.jpg
[image5]: ./test_images_output/lane_markings.jpg
[image6]: ./test_images_output/lane_markings_smooth.jpg

1. Image is converted to gray scale inorder to be used in canny edge detection algorithm.

### Raw Image
![alt text][image1]

### Grayscale
![grayscale][image2]

2. The image is filtered using Gaussian filter and then Canny edge detection algorithm is used to find the edges in the image.
### Edges
![canny edge][image3]

3. The image is then masked to show only region of interest where the lanes could be prasent. For performing this the "polyfill" function is used. The a trapezoidal region in the bottom half of the image is selected as this is the region where road is found.
### Masked Edges
![masked edge] [image4]

4. The hough tranform is performed in the masked edges image. Hough transform is used to find the line segments present in the masked edge image. Then these line segments are drwan on the orginal image. The line segments found by hough transform are the lane markings but there could also be some lines other than the lane markings. So these lines have to be averaged and extrapolated which is done in the next step.
### Lane Markings
![lane_markings] [image5]

