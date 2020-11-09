import numpy as np
import cv2
from camera import Camera
from camera_params.bus import *

if __name__ == "__main__":
    cam = Camera(RIGHT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION)
    image = cv2.imread("input/cam0.jpg")

    image1 = cam(image)
    # image1 = cv2.resize(image, TARGET_RESOLUTION)

    cam.fine_tine(0, 0, 10, 0, 0, 0, clear_mode=False)
    image2 = cam(image)
    cv2.imshow("test", np.concatenate((image1, image2), axis=1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("output/test.jpg", np.concatenate((image1, image2), axis=1))
