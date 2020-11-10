import argparse
import cv2
import numpy as np

from camera import Camera
from camera_params.bus import BACK, FRONT, LEFT, RIGHT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('-i', '--cap_id', type=int, default=0)
    args = parse.parse_args()

    cam_back = Camera(BACK, ORIGINAL_RESOLUTION, TARGET_RESOLUTION)
    cam_front = Camera(FRONT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION)
    cam_left = Camera(LEFT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION)
    cam_right = Camera(RIGHT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION)

    cap = cv2.VideoCapture(args.cap_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        _, frame = cap.read()
        back, front = cam_back.warp_distort(frame), cam_front.warp_distort(frame)
        left, right = cam_left.warp_distort(frame), cam_right.warp_distort(frame)
        res = np.concatenate((np.concatenate((back, front), axis=1), np.concatenate((left, right), axis=1)), axis=0)

        cv2.imshow("test", cv2.resize(res, (1920, 1080)))
        if cv2.waitKey(100) == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
    cv2.imwrite("output/test.jpg", res)
