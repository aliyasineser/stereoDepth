import numpy as np
import cv2
import sys
import glob

i = 0


def main():
    """ If you are autosync camera, you will get the images concatanated. This code is for seperation. """
    global i
    if len(sys.argv) < 4:
        print("Usage: ./program_name directory directory_to_save prefix_left_right_stereo")
        sys.exit(1)
    images = glob.glob(sys.argv[1] + '*.png')

    for image in images:
        img = cv2.imread(image)
        height, width, channels = img.shape
        print("for image name: " + image + " -> " + str(width) + " " + str(height) + "  " + str(channels))

    # Get the right image or left image as valid, possible to get both.
    if(sys.argv[3] == "right"):
        rightImage = img[0:height, width/2:width, :]
        cv2.imwrite(sys.argv[2] + "/" + sys.argv[3] + str(i)+".png", rightImage)
        i += 1
    elif(sys.argv[3] == "left"):
        leftImage = img[0:height, 0:width/2, :]
        cv2.imwrite(sys.argv[2] + "/" + sys.argv[3] + str(i)+".png", leftImage)
        i += 1
    else:
        leftImage = img[0:height, 0:width/2, :]
        rightImage = img[0:height, width/2:width, :]
        cv2.imwrite(sys.argv[2] + "/" + "right" + str(i)+".png", rightImage)
        cv2.imwrite(sys.argv[2] + "/" + "left" + str(i)+".png", leftImage)
        i += 1

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
