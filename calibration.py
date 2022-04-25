def camera_calibration():
    
    global camera_mtx,dist_coeff
    # Preparing the points for the object 
    objectPoints = np.zeros((6 * 9, 3), np.float32)
    objectPoints[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Storing all the points for the object and the image from all the images
    object_points = []
    image_points = []

    # Getting the directory of all of the calibrated images
    img = glob.glob('./camera_calibrations/*.jpg')
    images = None

    for indx, fname in enumerate(img):
        images = cv2.imread(fname)
        gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
        # it should be colored image or a 8-bit grayscale  
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret == True:
            object_points.append(objectPoints)
            image_points.append(corners)

    # getting the size of the Image 
    imageSize = (images.shape[1], images.shape[0])

    # Calibrate camera
    ret, camera_mtx, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, imageSize, None, None)

camera_calibration()
