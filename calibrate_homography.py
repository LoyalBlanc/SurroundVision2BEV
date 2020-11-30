import os

import cv2
import numpy as np

from camera_params.tiev_plus import BACK, FRONT, LEFT, RIGHT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION


def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        cv2.circle(img_res, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img_res, xy, (x - 50, y + 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", cv2.resize(img_res, (500, 500)))


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


img_name = "347"
script = """\"\"\"
相机参数：
视频流按CAMERA_ID顺序读取 分别为 Back, Front, Left, Right (字母序)
内参矩阵 K: [[fx,  s, x0],   # fxy 焦距
            [ 0, fy, y0],   # xy0 主点偏移
            [ 0,  0,  1]]   # 坐标轴倾斜参数s默认为0
畸变系数 D: 鱼眼相机 [k1, k2, k3, k4] -> 1 + k1*r^2 + k2*r^4 + k3*r^6 + k4*r^8
\"\"\"
import numpy as np\n
"""
# BACK
undist_img = un_dist(BACK, cv2.imread(f"./input/0_{img_name}.jpg"))
bb = np.mean(undist_img)
src_pts = np.array([[494, 364], [156, 370], [250, 302], [411, 295]])
dst_pts = np.array([[460, 678], [630, 680], [622, 738], [452, 742]])
for ptx, pty in src_pts:
    cv2.circle(undist_img, (ptx, pty), 1, (255, 0, 0), thickness=-1)
warp = cv2.getPerspectiveTransform(src_pts.astype(np.float32), dst_pts.astype(np.float32))
result = cv2.warpPerspective(undist_img, warp, TARGET_RESOLUTION)
size = 405
mask = np.concatenate((np.zeros([1000 - size, 1000, 3]), np.ones([size, 1000, 3])), axis=0)
back = np.where(mask, result, 0)
script += f"""
BACK = {'{'}
    "K": np.{BACK['K'].__repr__()},
    "D": np.{BACK['D'].__repr__()},
    "src": np.{src_pts.__repr__()},
    "dst": np.{dst_pts.__repr__()},
    "limit": np.concatenate((np.zeros([{1000-size}, 1000, 3]), np.ones([{size}, 1000, 3])), axis=0),
{'}'}
"""

# FRONT
undist_img = un_dist(FRONT, cv2.imread(f"./input/2_{img_name}.jpg"))
ff = np.mean(undist_img)
src_pts = np.array([[203, 271], [424, 274], [561, 319], [60, 315]])
dst_pts = np.array([[382, 55], [680, 66], [682, 133], [382, 133]])
for ptx, pty in src_pts:
    cv2.circle(undist_img, (ptx, pty), 1, (255, 0, 0), thickness=-1)
warp = cv2.getPerspectiveTransform(src_pts.astype(np.float32), dst_pts.astype(np.float32))
result = cv2.warpPerspective(undist_img, warp, TARGET_RESOLUTION)
size = 210
mask = np.concatenate((np.ones([size, 1000, 3]), np.zeros([1000 - size, 1000, 3])), axis=0)
front = np.where(mask, result, 0)
script += f"""
FRONT = {'{'}
    "K": np.{FRONT['K'].__repr__()},
    "D": np.{FRONT['D'].__repr__()},
    "src": np.{src_pts.__repr__()},
    "dst": np.{dst_pts.__repr__()},
    "limit": np.concatenate((np.ones([{size}, 1000, 3]), np.zeros([{1000-size}, 1000, 3])), axis=0),
{'}'}
"""

# LEFT
undist_img = un_dist(LEFT, cv2.imread(f"./input/4_{img_name}.jpg"))
ll = np.mean(undist_img)
src_pts = np.array([[537, 231], [622, 264], [142, 212], [217, 200]])
dst_pts = np.array([[160, 81], [240, 80], [378, 748], [270, 732]])
for ptx, pty in src_pts:
    cv2.circle(undist_img, (ptx, pty), 1, (255, 0, 0), thickness=-1)
warp = cv2.getPerspectiveTransform(src_pts.astype(np.float32), dst_pts.astype(np.float32))
result = cv2.warpPerspective(undist_img, warp, TARGET_RESOLUTION)
size = 485
mask = np.concatenate((np.ones([1000, size, 3]), np.zeros([1000, 1000 - size, 3])), axis=1)
left = np.where(mask, result, 0)
script += f"""
LEFT = {'{'}
    "K": np.{LEFT['K'].__repr__()},
    "D": np.{LEFT['D'].__repr__()},
    "src": np.{src_pts.__repr__()},
    "dst": np.{dst_pts.__repr__()},
    "limit": np.concatenate((np.ones([1000, {size}, 3]), np.zeros([1000, {1000-size}, 3])), axis=1),
{'}'}
"""

# RIGHT
undist_img = un_dist(RIGHT, cv2.imread(f"./input/6_{img_name}.jpg"))
rr = np.mean(undist_img)
src_pts = np.array([[236, 231], [223, 208], [353, 190], [450, 197]])
dst_pts = np.array([[825, 331], [990, 320], [888, 728], [685, 772]])
for ptx, pty in src_pts:
    cv2.circle(undist_img, (ptx, pty), 1, (255, 0, 0), thickness=-1)
warp = cv2.getPerspectiveTransform(src_pts.astype(np.float32), dst_pts.astype(np.float32))
result = cv2.warpPerspective(undist_img, warp, TARGET_RESOLUTION)
size = 345
mask = np.concatenate((np.zeros([1000, 1000 - size, 3]), np.ones([1000, size, 3])), axis=1)
right = np.where(mask, result, 0)
script += f"""
RIGHT = {'{'}
    "K": np.{RIGHT['K'].__repr__()},
    "D": np.{RIGHT['D'].__repr__()},
    "src": np.{src_pts.__repr__()},
    "dst": np.{dst_pts.__repr__()},
    "limit": np.concatenate((np.zeros([1000, {1000-size}, 3]), np.ones([1000, {size}, 3])), axis=1),
{'}'}

ORIGINAL_RESOLUTION = {ORIGINAL_RESOLUTION}
TARGET_RESOLUTION = {TARGET_RESOLUTION}
"""

bf = cv2.bitwise_or(back, front)
lr = cv2.bitwise_or(left, right)
img_res = cv2.bitwise_or(bf, lr)

cv2.namedWindow("image")
cv2.setMouseCallback("image", click)
cv2.imshow("image", img_res)
cv2.waitKey(0)
cv2.destroyAllWindows()

os.makedirs("./output/", exist_ok=True)
with open('output/warp_homography.txt', 'w+') as f:
    f.write(str(script))
