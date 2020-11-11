import cv2

from camera import Camera
from camera_params.bus import BACK, FRONT, LEFT, RIGHT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION

if __name__ == "__main__":
    # cam_back = Camera(BACK, ORIGINAL_RESOLUTION, TARGET_RESOLUTION)
    cam_front = Camera(FRONT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION)
    # cam_left = Camera(LEFT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION)
    # cam_right = Camera(RIGHT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION)

    image = cv2.resize(cv2.imread("input/bus/front1.jpg"), ORIGINAL_RESOLUTION)
    cam_front.fine_tining(image)
