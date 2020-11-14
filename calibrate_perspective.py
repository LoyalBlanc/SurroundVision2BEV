import cv2
import numpy as np

from camera import Camera
from camera_params.tiev_plus import BACK, FRONT, LEFT, RIGHT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION

# B F L R
cam_list = [
    Camera(BACK, ORIGINAL_RESOLUTION, TARGET_RESOLUTION),
    Camera(FRONT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION),
    Camera(LEFT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION),
    Camera(RIGHT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION),
]
mask_list = [
    np.concatenate((np.zeros([TARGET_RESOLUTION[1] // 2, TARGET_RESOLUTION[0], 3]),
                    np.ones([TARGET_RESOLUTION[1] // 2, TARGET_RESOLUTION[0], 3])), axis=0),
    np.concatenate((np.ones([TARGET_RESOLUTION[1] // 2 - 100, TARGET_RESOLUTION[0], 3]),
                    np.zeros([TARGET_RESOLUTION[1] // 2 + 100, TARGET_RESOLUTION[0], 3])), axis=0),
    np.concatenate((np.ones([TARGET_RESOLUTION[1], TARGET_RESOLUTION[0] // 2, 3]),
                    np.zeros([TARGET_RESOLUTION[1], TARGET_RESOLUTION[0] // 2, 3])), axis=1),
    np.concatenate((np.zeros([TARGET_RESOLUTION[1], TARGET_RESOLUTION[0] // 2 + 120, 3]),
                    np.ones([TARGET_RESOLUTION[1], TARGET_RESOLUTION[0] // 2 - 120, 3])), axis=1) * np.concatenate(
        (np.zeros([TARGET_RESOLUTION[1] // 2 - 240, TARGET_RESOLUTION[0], 3]),
         np.ones([TARGET_RESOLUTION[1] // 2 + 240, TARGET_RESOLUTION[0], 3])), axis=0),
]

ori_index = ""
ori_list = [
    cv2.resize(cv2.imread(f"input/tiev_plus/back{ori_index}.jpg"), ORIGINAL_RESOLUTION),
    cv2.resize(cv2.imread(f"input/tiev_plus/front{ori_index}.jpg"), ORIGINAL_RESOLUTION),
    cv2.resize(cv2.imread(f"input/tiev_plus/left{ori_index}.jpg"), ORIGINAL_RESOLUTION),
    cv2.resize(cv2.imread(f"input/tiev_plus/right{ori_index}.jpg"), ORIGINAL_RESOLUTION),
]
img_list = [np.where(mask, cam(ori), 0) for cam, mask, ori in zip(cam_list, mask_list, ori_list)]

flag = 0
while True:
    key = cv2.waitKey(0)
    if key == 27:
        break
    elif 48 < key < 53:
        flag = key - 49

    img_list[flag] = np.where(mask_list[flag], cam_list[flag].fine_tining(ori_list[flag], key), 0)
    img_bf = img_list[0] + img_list[1]
    img_lr = img_list[2] + img_list[3]
    img_res = np.where(img_lr > 10, np.where(img_bf > 10, img_lr // 2 + img_bf // 2, img_lr), img_bf)
    cv2.imshow("fine_tining", cv2.resize(img_res, (1080, 768)))  # TARGET_RESOLUTION))

cv2.destroyAllWindows()

script = f"""import numpy as np
 
BACK = {'{'}
    "K": np.{BACK['K'].__repr__()},
    "D": np.{BACK['D'].__repr__()},
    "T": np.{cam_list[0].transform.__repr__()}
{'}'}
FRONT = {'{'}
    "K": np.{FRONT['K'].__repr__()},
    "D": np.{FRONT['D'].__repr__()},
    "T": np.{cam_list[1].transform.__repr__()}
{'}'}
LEFT = {'{'}
    "K": np.{LEFT['K'].__repr__()},
    "D": np.{LEFT['D'].__repr__()},
    "T": np.{cam_list[2].transform.__repr__()}
{'}'}
RIGHT = {'{'}
    "K": np.{RIGHT['K'].__repr__()},
    "D": np.{RIGHT['D'].__repr__()},
    "T": np.{cam_list[3].transform.__repr__()}
{'}'}

ORIGINAL_RESOLUTION = {ORIGINAL_RESOLUTION}
TARGET_RESOLUTION = {TARGET_RESOLUTION}
"""

with open('output/warp_perspective.txt', 'w+') as f:
    f.write(str(script))
cv2.imwrite("output/result.jpg", img_res)
