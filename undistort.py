def undistort(images, camera_mtx, dist_coeff):
    """
    we will use cv2.undistort to undistort
    :param images: we will assume that the input image is RGB (imread by mpimg)
    :param mtx: it the parameter for the calibration of the camera 
    :param dist:is a calibration parameter for the camera
    :return: Undistorted image
    """
   
    undistortedImage = cv2.undistort(images, camera_mtx, dist_coeff, None, mtx)

    return undistorted_img