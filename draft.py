import cv2
import numpy as np

from camera_params.tiev_plus import BACK, FRONT, LEFT, RIGHT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION

src_pts = np.float32([

])
dst_pts = np.float32([

])

warp = cv2.findHomography(src_pts, dst_pts)

img = cv2.imread("")
cam = BACK
camera_matrix, dist_coefficient = cam['K'], cam['D']
target_co = 2

camera_target = np.array(camera_matrix)
camera_target[0][0] = camera_target[0][0] / 4
camera_target[0][2] = ORIGINAL_RESOLUTION[0] * target_co / 2
camera_target[1][1] = camera_target[1][1] / 4
camera_target[1][2] = ORIGINAL_RESOLUTION[1] * target_co / 2
target_size = (int(ORIGINAL_RESOLUTION[0] * target_co),
               int(ORIGINAL_RESOLUTION[1] * target_co))

map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    camera_matrix, dist_coefficient, np.eye(3),
    camera_target, target_size, cv2.CV_32F)

undist_img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
cv2.imshow("undist", undist_img)
cv2.waitKey(0)

rectify_x = cv2.warpPerspective(map1, warp, TARGET_RESOLUTION)
rectify_y = cv2.warpPerspective(map2, warp, TARGET_RESOLUTION)

pers_img = cv2.remap(undist_img, rectify_x, rectify_y, cv2.INTER_LINEAR)
cv2.imshow("pers", pers_img)
cv2.waitKey(0)

cv2.destroyAllWindows()
