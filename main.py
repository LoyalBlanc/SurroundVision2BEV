import cv2
import numpy as np

from camera import Camera
from camera_params.bus import BACK, FRONT, LEFT, RIGHT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION

if __name__ == "__main__":
    cam_back = Camera(BACK, ORIGINAL_RESOLUTION, TARGET_RESOLUTION)
    cam_front = Camera(FRONT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION)
    cam_left = Camera(LEFT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION)
    cam_right = Camera(RIGHT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION)

    back = cam_back.warp_distort(cv2.resize(cv2.imread("input/bus/back1.jpg"), ORIGINAL_RESOLUTION))
    front = cam_front.warp_distort(cv2.resize(cv2.imread("input/bus/front1.jpg"), ORIGINAL_RESOLUTION))

    left = cam_left.warp_distort(cv2.resize(cv2.imread("input/bus/left1.jpg"), ORIGINAL_RESOLUTION))
    right = cam_right.warp_distort(cv2.resize(cv2.imread("input/bus/right1.jpg"), ORIGINAL_RESOLUTION))

    res = np.concatenate((np.concatenate((back, front), axis=1), np.concatenate((left, right), axis=1)), axis=0)

    cv2.imshow("test", cv2.resize(res, (1920, 1080)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
