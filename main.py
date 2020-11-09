import cv2
from camera import Camera

if __name__ == "__main__":
    from camera_params.bus import *

    cam = Camera(RIGHT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION)
    image = cv2.imread("input/cam3.jpg")

    image1 = cam(image)

    cam.fine_tine(x=0.1, y=0.1, z=0.1, pitch=0.1, yaw=0.1)
    image2 = cam(image)
    cv2.imwrite("output/test.jpg", np.concatenate((image1, image2), axis=1))
