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
    np.concatenate((np.zeros([TARGET_RESOLUTION[1] // 2 + 40, TARGET_RESOLUTION[0], 3]),
                    np.ones([TARGET_RESOLUTION[1] // 2 - 40, TARGET_RESOLUTION[0], 3])), axis=0),
    np.concatenate((np.ones([TARGET_RESOLUTION[1] // 2 - 40, TARGET_RESOLUTION[0], 3]),
                    np.zeros([TARGET_RESOLUTION[1] // 2 + 40, TARGET_RESOLUTION[0], 3])), axis=0),
    np.concatenate((np.ones([TARGET_RESOLUTION[1], TARGET_RESOLUTION[0] // 2 - 40, 3]),

                    np.zeros([TARGET_RESOLUTION[1], TARGET_RESOLUTION[0] // 2 + 40, 3])), axis=1) * np.concatenate(
        (np.zeros([TARGET_RESOLUTION[1] // 2 - 240, TARGET_RESOLUTION[0], 3]),
         np.ones([TARGET_RESOLUTION[1] // 2 + 240, TARGET_RESOLUTION[0], 3])), axis=0),

    np.concatenate((np.zeros([TARGET_RESOLUTION[1], TARGET_RESOLUTION[0] // 2 + 80, 3]),
                    np.ones([TARGET_RESOLUTION[1], TARGET_RESOLUTION[0] // 2 - 80, 3])), axis=1) * np.concatenate(
        (np.zeros([TARGET_RESOLUTION[1] // 2 - 120, TARGET_RESOLUTION[0], 3]),
         np.ones([TARGET_RESOLUTION[1] // 2 + 120, TARGET_RESOLUTION[0], 3])), axis=0)
]

ori_index = "597"
ori_list = [
    cv2.resize(cv2.imread(f"input/{ori_index}/0_{ori_index}.jpg"), ORIGINAL_RESOLUTION),
    cv2.resize(cv2.imread(f"input/{ori_index}/2_{ori_index}.jpg"), ORIGINAL_RESOLUTION),
    cv2.resize(cv2.imread(f"input/{ori_index}/4_{ori_index}.jpg"), ORIGINAL_RESOLUTION),
    cv2.resize(cv2.imread(f"input/{ori_index}/6_{ori_index}.jpg"), ORIGINAL_RESOLUTION),
]
img_list = [np.where(mask, cam(ori), 0) for cam, mask, ori in zip(cam_list, mask_list, ori_list)]

ori_index2 = "651"
ori_list2 = [
    cv2.resize(cv2.imread(f"input/{ori_index2}/0_{ori_index2}.jpg"), ORIGINAL_RESOLUTION),
    cv2.resize(cv2.imread(f"input/{ori_index2}/2_{ori_index2}.jpg"), ORIGINAL_RESOLUTION),
    cv2.resize(cv2.imread(f"input/{ori_index2}/4_{ori_index2}.jpg"), ORIGINAL_RESOLUTION),
    cv2.resize(cv2.imread(f"input/{ori_index2}/6_{ori_index2}.jpg"), ORIGINAL_RESOLUTION),
]
img_list2 = [np.where(mask, cam(ori), 0) for cam, mask, ori in zip(cam_list, mask_list, ori_list2)]

flag = 0
while True:
    key = cv2.waitKey(0)
    if key == 27:
        break
    elif 48 < key < 53:
        flag = key - 49

    cam_list[flag].fine_tining(key)

    img_list[flag] = np.where(mask_list[flag], cam_list[flag](ori_list[flag]), 0)
    img_bf = np.where(img_list[0] > 10, img_list[0], np.where(img_list[1] > 10, img_list[1], 0))
    img_lr = np.where(img_list[2] > 10, img_list[2], np.where(img_list[3] > 10, img_list[3], 0))
    img_res = np.where(img_lr > 10, np.where(img_bf > 10, img_bf // 2 + img_lr // 2, img_lr), img_bf)

    # cv2.imshow("fine_tining", img_res)  # cv2.resize(img_res, (1000, 1000)))  # TARGET_RESOLUTION))

    img_list2[flag] = np.where(mask_list[flag], cam_list[flag](ori_list2[flag]), 0)
    img_bf2 = np.where(img_list2[0] > 10, img_list2[0], np.where(img_list2[1] > 10, img_list2[1], 0))
    img_lr2 = np.where(img_list2[2] > 10, img_list2[2], np.where(img_list2[3] > 10, img_list2[3], 0))
    img_res2 = np.where(img_lr2 > 10, np.where(img_bf2 > 10, img_bf2 // 2 + img_lr2 // 2, img_lr2), img_bf2)

    cv2.imshow("fine_tining2", np.concatenate((img_res, img_res2), axis=1))

cv2.destroyAllWindows()
# mask = (img_list[3] > 10).astype(np.int8)
# cv2.imwrite("test.jpg", mask * 255)
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
