import cv2

from camera import Camera
from camera_params.bus import BACK, FRONT, LEFT, RIGHT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION

if __name__ == "__main__":
    cam_list = [
        Camera(BACK, ORIGINAL_RESOLUTION, TARGET_RESOLUTION),
        Camera(FRONT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION),
        Camera(LEFT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION),
        Camera(RIGHT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION),
    ]
    ori_list = [
        cv2.resize(cv2.imread("input/bus/back2.jpg"), ORIGINAL_RESOLUTION),
        cv2.resize(cv2.imread("input/bus/front2.jpg"), ORIGINAL_RESOLUTION),
        cv2.resize(cv2.imread("input/bus/left2.jpg"), ORIGINAL_RESOLUTION),
        cv2.resize(cv2.imread("input/bus/right2.jpg"), ORIGINAL_RESOLUTION),
    ]
    img_list = [cam(ori) for cam, ori in zip(cam_list, ori_list)]

    flag = 0
    while True:
        key = cv2.waitKey(0)
        if key == 27:
            break
        elif 48 < key < 53:
            flag = key - 49

        img_list[flag] = cam_list[flag].fine_tining(ori_list[flag], key)
        img_res = img_list[0] + img_list[1] + img_list[2] + img_list[3]
        cv2.imshow("fine_tining", cv2.resize(img_res, TARGET_RESOLUTION))

    cv2.destroyAllWindows()
    with open('output/warp_perspective.txt', 'w+') as f:
        script = f""" "T": np.{cam_list[0].transform.__repr__()}\n "T": np.{cam_list[1].transform.__repr__()}\n""" \
                 f""" "T": np.{cam_list[2].transform.__repr__()}\n "T": np.{cam_list[3].transform.__repr__()}\n"""
        f.write(str(script))
