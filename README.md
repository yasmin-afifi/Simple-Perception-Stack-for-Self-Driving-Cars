# Lane Line Finding Project 

**The steps of this project are the following:**

1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2. Apply a distortion correction to raw images.
3. Use color transforms, gradients, etc., to create a thresholded binary image.
4. Apply a perspective transform to rectify binary image ("birds-eye view").
5. Detect lane pixels and fit to find the lane boundary.
6. Determine the curvature of the lane and vehicle position with respect to center.
7. Warp the detected lane boundaries back onto the original image.
8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
# 1. Calibration and Distorion Correction:
The chessboard corners are the reference to generate objpoints and imgpoints.
I then used the output objpoints and imgpoints to compute the camera calibration and distortion coefficients using the cv2.calibrateCamera() function.
Matrix mtx and dist from camera calibration are applied to distortion correction to one of the test images like this one:

# 2.Prespective and inverse prespective Transform
Function perspective_transform() performs perspective transform given source and destination points.
# 3. Thresholding:
differences in RGB space do not correspond well to perceived differences in color. 
That is, two colors can be close in RGB space but appear very different to humans and vice versa.
LUV decouple the "color" (chromaticity, the UV part) and "lightness" (luminance, the L part) of color.
Thus in object detection, it is common to match objects just based on the UV part,
which gives invariance to changes in lighting condition.
LAB is designed to approximate human vision. The L* component closely matches human perception of lightnessis useful for predicting small differences in color.So we will use B channel from LAB space identified yellow lanes while L channel from LUV space could detect white lanes
# 4. Lane Detection and polynomial Fit
Firstly, We calculated the histogram of non-zero x-axis in binary image. And based on the maximum sum of x position, We used sliding window search method to identify lane pixels. If it is previously detected in last frame, a quick search could be applied based on last detected x/y pixel positions with a proper margin. Then I fitted lane lines with a 2nd order polynomial
# 5.Curvature and position Detection 
To connect pixel unit with real world meter unit, We defined conversions in x and y from pixels space to meters. In order to calculate precisely, We used detected lane width dynamically.
# 6. Back Transformation
Last step is to transform the lane identified back onto the road image.
