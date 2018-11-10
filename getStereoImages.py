import numpy as np
import cv2
import glob
import argparse
import sys

# Set the values for your cameras
capL = cv2.VideoCapture(1)
capR = cv2.VideoCapture(0)

# Use these if you need high resolution.
# capL.set(3, 1024) # width
# capL.set(4, 768) # height

# capR.set(3, 1024) # width
# capR.set(4, 768) # height
i = 0


def main():
    global i
    if len(sys.argv) < 3:
        print("Usage: ./program_name directory_to_save start_index")
        sys.exit(1)

    i = int(sys.argv[2])  # Get the start number.

    while True:
        # Grab and retreive for sync
        if not (capL.grab() and capR.grab()):
            print("No more frames")
            break

        _, leftFrame = capL.retrieve()
        _, rightFrame = capR.retrieve()

        # Use if you need high resolution. If you set the camera for high res, you can pass these.
        # cv2.namedWindow('capL', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('capL', 1024, 768)

        # cv2.namedWindow('capR', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('capR', 1024, 768)

        cv2.imshow('capL', leftFrame)
        cv2.imshow('capR', rightFrame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            cv2.imwrite(sys.argv[1] + "/left" + str(i) + ".png", leftFrame)
            cv2.imwrite(sys.argv[1] + "/right" + str(i) + ".png", rightFrame)
            i += 1

    capL.release()
    capR.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
