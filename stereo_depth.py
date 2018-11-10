import numpy as np
import cv2
import argparse
import sys
from calibration_store import load_stereo_coefficients

def depth_map(imgL, imgR):
    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
    # SGBM Parameters -----------------
    window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=5*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.3
    visual_multiplier = 6

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)

    return filteredImg


if __name__ == '__main__':
    # Args handling -> check help parameters to understand
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument('--calibration_file', type=str, required=True, help='Path to the stereo calibration file')
    parser.add_argument('--left_source', type=str, required=True, help='Left video or v4l2 device name')
    parser.add_argument('--right_source', type=str, required=True, help='Right video or v4l2 device name')
    parser.add_argument('--is_real_time', type=int, required=True, help='Is it camera stream or video')

    args = parser.parse_args()

    # is camera stream or video
    if args.is_real_time:
        cap_left = cv2.VideoCapture(args.left_source, cv2.CAP_V4L2)
        cap_right = cv2.VideoCapture(args.right_source, cv2.CAP_V4L2)
    else:
        cap_left = cv2.VideoCapture(args.left_source)
        cap_right = cv2.VideoCapture(args.right_source)

    K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q = load_stereo_coefficients(args.calibration_file)  # Get cams params

    if not cap_left.isOpened() and not cap_right.isOpened():  # If we can't get images from both sources, error
        print("Can't opened the streams!")
        sys.exit(-9)

    # Change the resolution in need
    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # float
    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # float

    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # float
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # float

    while True:  # Loop until 'q' pressed or stream ends
        # Grab&retreive for sync images
        if not (cap_left.grab() and cap_right.grab()):
            print("No more frames")
            break

        _, leftFrame = cap_left.retrieve()
        _, rightFrame = cap_right.retrieve()
        height, width, channel = leftFrame.shape  # We will use the shape for remap

        # Undistortion and Rectification part!
        leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (width, height), cv2.CV_32FC1)
        left_rectified = cv2.remap(leftFrame, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (width, height), cv2.CV_32FC1)
        right_rectified = cv2.remap(rightFrame, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

        # We need grayscale for disparity map.
        gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

        disparity_image = depth_map(gray_left, gray_right)  # Get the disparity map

        # Show the images
        cv2.imshow('left(R)', leftFrame)
        cv2.imshow('right(R)', rightFrame)
        cv2.imshow('Disparity', disparity_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Get key to stop stream. Press q for exit
            break

    # Release the sources.
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()
