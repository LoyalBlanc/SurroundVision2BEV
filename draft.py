import cv2

from module.camera import Camera
from avp3.camera_param import BACK, FRONT, LEFT, RIGHT, ORIGINAL_RESOLUTION

if __name__ == "__main__":
    image = cv2.imread("input/tr.jpg")
    cam = Camera(BACK, ORIGINAL_RESOLUTION, (1080, 1920))
    cv2.imwrite("output/back.jpg", cam(image))

    image = cv2.imread("input/tl.jpg")
    cam = Camera(FRONT, ORIGINAL_RESOLUTION, (540, 960))
    cv2.imwrite("output/front.jpg", cam(image))

    image = cv2.imread("input/bl.jpg")
    cam = Camera(LEFT, ORIGINAL_RESOLUTION, (540, 960))
    cv2.imwrite("output/left.jpg", cam(image))

    image = cv2.imread("input/br.jpg")
    cam = Camera(RIGHT, ORIGINAL_RESOLUTION, (540, 960))
    cv2.imwrite("output/right.jpg", cam(image))
