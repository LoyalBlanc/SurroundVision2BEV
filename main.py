import numpy as np
import cv2
from camera import Camera
from camera_params.bus import *

if __name__ == "__main__":
    image = cv2.imread("input/bus/back1.jpg")

    back = Camera(BACK, ORIGINAL_RESOLUTION, TARGET_RESOLUTION)(image)
    front = Camera(FRONT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION)(image)
    left = Camera(LEFT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION)(image)
    right = Camera(RIGHT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION)(image)

    res1 = np.concatenate((back, front), axis=1)
    res2 = np.concatenate((left, right), axis=1)
    res = np.concatenate((res1, res2), axis=0)

    cv2.imshow("test", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("output/test.jpg", res)
