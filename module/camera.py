import cv2
import numpy as np


class Camera(object):
    def __init__(self, camera_param, original_resolution, target_resolution):
        camera_matrix, dist_coefficient, trans_matrix = camera_param['K'], camera_param['D'], camera_param['M']

        camera_target = np.array([[100., 0., original_resolution[0] / 2.],
                                  [0., 100., original_resolution[1] / 2.],
                                  [0., 0., 1.]])

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix, dist_coefficient, np.eye(3),
            camera_target, original_resolution, cv2.CV_32F)

        map1 = cv2.warpPerspective(map1, trans_matrix, (original_resolution, original_resolution))
        self.map1 = cv2.resize(map1, target_resolution)

        map2 = cv2.warpPerspective(map2, trans_matrix, (original_resolution, original_resolution))
        self.map2 = cv2.resize(map2, target_resolution)

    def __call__(self, img):
        return cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)
