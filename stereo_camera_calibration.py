import numpy as np
import cv2
import glob
import argparse
import sys
from calibration_store import load_coefficients, save_stereo_coefficients

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
image_size = None

def stereo_calibrate(left_file, right_file, left_dir, left_prefix, right_dir, right_prefix, image_format, save_file, square_size, width=9, height=6):
    """ Stereo calibration and rectification """
    objp, leftp, rightp = load_image_points(left_dir, left_prefix, right_dir, right_prefix, image_format, square_size, width, height)

    K1, D1 = load_coefficients(left_file)
    K2, D2 = load_coefficients(right_file)

    flag = 0
    #flag |= cv2.CALIB_FIX_INTRINSIC
    flag |= cv2.CALIB_USE_INTRINSIC_GUESS
    ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objp, leftp, rightp, K1, D1, K2, D2, image_size)
    print("Stereo calibration rms: ", ret)
    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9)

    save_stereo_coefficients(save_file, K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q)


def load_image_points(left_dir, left_prefix, right_dir, right_prefix, image_format, square_size, width=9, height=6):
    global image_size
    pattern_size = (width, height)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    left_imgpoints = []  # 2d points in image plane.
    right_imgpoints = []  # 2d points in image plane.

    if left_dir[-1:] == '/':
        left_dir = left_dir[:-1]

    if right_dir[-1:] == '/':
        right_dir = right_dir[:-1]

    left_images = glob.glob(left_dir + '/' + left_prefix + '*.' + image_format)
    right_images = glob.glob(right_dir + '/' + right_prefix + '*.' + image_format)

    left_images.sort()
    right_images.sort()

    if len(left_images) != len(right_images):
        print("Numbers of left and right images are not equal. They should be pairs.")
        print("Left images count: ", len(left_images))
        print("Right images count: ", len(right_images))
        sys.exit(-1)

    pair_images = zip(left_images, right_images)


    for left_im, right_im in pair_images:
        # Right Object Points
        right = cv2.imread(right_im)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, pattern_size,
                                                             cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)

        # Left Object Points
        left = cv2.imread(left_im)
        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, pattern_size,
                                                           cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)

        if ret_left and ret_right:
            # Object points
            objpoints.append(objp)
            # Right points
            corners2_right = cv2.cornerSubPix(gray_right, corners_right, (5, 5), (-1, -1), criteria)
            right_imgpoints.append(corners2_right)
            # Left points
            corners2_left = cv2.cornerSubPix(gray_left, corners_left, (5, 5), (-1, -1), criteria)
            left_imgpoints.append(corners2_left)
        else:
            print("Chessboard couldn't detected. Image pair: ", left_im, " and ", right_im)
            continue


    image_size = gray_right.shape
    return [objpoints, left_imgpoints, right_imgpoints]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument('--left_file', type=str, required=True, help='left matrix file')
    parser.add_argument('--right_file', type=str, required=True, help='right matrix file')
    parser.add_argument('--left_prefix', type=str, required=True, help='left image prefix')
    parser.add_argument('--right_prefix', type=str, required=True, help='right image prefix')
    parser.add_argument('--left_dir', type=str, required=True, help='left images directory path')
    parser.add_argument('--right_dir', type=str, required=True, help='right images directory path')
    parser.add_argument('--image_format', type=str, required=True, help='image format, png/jpg')
    parser.add_argument('--width', type=int, required=False, help='chessboard width size, default is 9')
    parser.add_argument('--height', type=int, required=False, help='chessboard height size, default is 6')
    parser.add_argument('--square_size', type=float, required=False, help='chessboard square size')
    parser.add_argument('--save_file', type=str, required=True, help='YML file to save stereo calibration matrices')

    args = parser.parse_args()
    if args.width == None and args.height == None:
        stereo_calibrate(args.left_file, args.right_file, args.left_dir, args.left_prefix, args.right_dir, args.right_prefix, args.image_format, args.save_file, args.square_size)
    else:
        stereo_calibrate(args.left_file, args.right_file, args.left_dir, args.left_prefix, args.right_dir, args.right_prefix, args.image_format, args.save_file, args.square_size, args.width, args.height)
