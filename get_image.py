import os

import cv2
import numpy as np
from time import time

from camera import Camera
from camera_params.tiev_plus import BACK, FRONT, LEFT, RIGHT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION

os.makedirs("./output/", exist_ok=True)
C_range = ('back', 'front', 'left', 'right')
cap_list = [cv2.VideoCapture(f"/dev/cam_{name}") for name in C_range]
cam_list = [
    Camera(BACK, ORIGINAL_RESOLUTION, TARGET_RESOLUTION),
    Camera(FRONT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION),
    Camera(LEFT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION),
    Camera(RIGHT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION),
]

while True:
    frame_list = [cap.read()[1] for cap in cap_list]
    # bf = np.concatenate((frame_list[0], frame_list[1]), axis=0)
    # lr = np.concatenate((frame_list[2], frame_list[3]), axis=0)
    # cv2.imshow("raw_frame", cv2.resize(np.concatenate((bf, lr), axis=1), TARGET_RESOLUTION))

    undist_list = [cam.warp_distort(frame) for cam, frame in zip(cam_list, frame_list)]
    bf = np.concatenate((undist_list[0], undist_list[1]), axis=0)
    lr = np.concatenate((undist_list[2], undist_list[3]), axis=0)
    cv2.imshow("un_dist", cv2.resize(np.concatenate((bf, lr), axis=1), TARGET_RESOLUTION))

    key = cv2.waitKey(100)
    if key == 120:
        stamp = int(time()) % 1000
        for name, frame in zip(C_range, frame_list):
            cv2.imwrite(f"output/{name}_{stamp}.jpg", frame)
    elif key == 122:
        break

cv2.destroyAllWindows()
for cap in cap_list:
    cap.release()
