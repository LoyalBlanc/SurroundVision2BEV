import cv2
import numpy as np


class Camera(object):
    def __init__(self, camera_param, original_resolution, target_resolution):
        camera_matrix, dist_coefficient = camera_param['K'], camera_param['D']

        if 'M' in camera_param.keys():
            trans_matrix = camera_param['M']
        else:
            trans_matrix = cv2.getPerspectiveTransform(camera_param['src'], camera_param['tag'])

        camera_target = np.array([[100., 0., original_resolution[0] / 2.],
                                  [0., 100., original_resolution[1] / 2.],
                                  [0., 0., 1.]])

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix, dist_coefficient, np.eye(3),
            camera_target, original_resolution, cv2.CV_32F)
        self.map1 = cv2.warpPerspective(cv2.resize(map1, target_resolution), trans_matrix, target_resolution)
        self.map2 = cv2.warpPerspective(cv2.resize(map2, target_resolution), trans_matrix, target_resolution)

        self.mask = camera_param['limit']

    def __call__(self, img):
        img = cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)
        return np.where(self.mask, img, 0)
