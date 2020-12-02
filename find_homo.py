import os

import cv2
import numpy as np

from camera_params.bus import BACK, FRONT, LEFT, RIGHT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION


def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        cv2.circle(img_res, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img_res, xy, (x - 50, y + 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img_res)


def un_dist(cam, img):
    camera_matrix, dist_coefficient = cam['K'], cam['D']

    camera_target = np.array(camera_matrix)
    camera_target[0][0] = camera_target[0][0] / 2
    camera_target[0][2] = ORIGINAL_RESOLUTION[0] / 2
    camera_target[1][1] = camera_target[1][1] / 2
    camera_target[1][2] = ORIGINAL_RESOLUTION[1] / 2

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        camera_matrix, dist_coefficient, np.eye(3),
        camera_target, ORIGINAL_RESOLUTION, cv2.CV_32F)
    return cv2.remap(img, map1, map2, cv2.INTER_LINEAR)


img = cv2.resize(cv2.imread(f"input/2_3.jpg"), ORIGINAL_RESOLUTION)
img_res = un_dist(LEFT, img)

cv2.namedWindow("image")
cv2.setMouseCallback("image", click)
cv2.imshow("image", img_res)
cv2.waitKey(0)
cv2.destroyAllWindows()
