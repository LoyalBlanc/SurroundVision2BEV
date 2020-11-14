import cv2
import numpy as np


class FineTuningParam(object):
    def __init__(self):
        self.param = {
            'x': [0, [97, 100]],
            'y': [0, [119, 115]],
            'z': [0, [113, 101]],
            'roll': [0, [107, 105]],
            'pitch': [0, [108, 106]],
            'yaw': [0, [117, 111]],
        }
        self.step = [0.1, [122, 120]]

    def update(self, key):
        for _, value in self.param.items():
            if key in value[1]:
                value[0] += -self.step[0] if key == value[1][0] else self.step[0]
                break

        if key in self.step[1]:
            self.step[0] *= 0.1 if key == self.step[1][0] else 10

        roll_rotation = np.matrix([[1, 0, 0],
                                   [0, np.cos(self.param['roll'][0]), -np.sin(self.param['roll'][0])],
                                   [0, np.sin(self.param['roll'][0]), np.cos(self.param['roll'][0])]])
        pitch_rotation = np.matrix([[np.cos(self.param['pitch'][0]), 0, -np.sin(self.param['pitch'][0])],
                                    [0, 1, 0],
                                    [np.sin(self.param['pitch'][0]), 0, np.cos(self.param['pitch'][0])]])
        yaw_rotation = np.matrix([[np.cos(self.param['yaw'][0]), -np.sin(self.param['yaw'][0]), 0],
                                  [np.sin(self.param['yaw'][0]), np.cos(self.param['yaw'][0]), 0],
                                  [0, 0, 1]])

        rotation = roll_rotation * pitch_rotation * yaw_rotation
        translation = np.matrix([[self.param['x'][0]], [self.param['y'][0]], [self.param['z'][0]]])
        new_transform = np.concatenate((np.concatenate((rotation, translation), axis=1),
                                        np.matrix([0, 0, 0, 1])), axis=0)

        return new_transform

    def reset(self):
        for _, value in self.param.items():
            value[0] = 0


class Camera(object):
    def __init__(self, camera_param, original_resolution, target_resolution, target_co=1):
        """
        :param camera_param: BACK = {
            "K": np.matrix 3x3,
            "D": np.matrix 1x4,
            "T": np.matrix 4x4
            }
        :param original_resolution: tuple(int, int)
        :param target_resolution: tuple(int, int)
        """
        self.camera_matrix = camera_param['K']
        self.dist_coefficient = camera_param['D']
        self.transform = camera_param['T']
        self.original_resolution = original_resolution
        self.target_resolution = target_resolution

        self.target_co = target_co

        self.warp = self.calculate_homogeneous()
        self.map = self.calculate_rectify()

        self.rectify_x = cv2.warpPerspective(self.map[0], self.warp, self.target_resolution)
        self.rectify_y = cv2.warpPerspective(self.map[1], self.warp, self.target_resolution)
        self.fine_tining_param = FineTuningParam()

    def __call__(self, img):
        return cv2.remap(img, self.rectify_x, self.rectify_y, cv2.INTER_LINEAR)

    def calculate_rectify(self):
        camera_target = np.array(self.camera_matrix)
        camera_target[0][0] = camera_target[0][0] / 2
        camera_target[0][2] = self.original_resolution[0] * self.target_co / 2
        camera_target[1][1] = camera_target[1][1] / 2
        camera_target[1][2] = self.original_resolution[1] * self.target_co / 2
        target_size = (int(self.original_resolution[0] * self.target_co),
                       int(self.original_resolution[1] * self.target_co))
        return cv2.fisheye.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coefficient, np.eye(3),
            camera_target, target_size, cv2.CV_32F)

    def warp_distort(self, img):
        return cv2.remap(img, self.map[0], self.map[1], cv2.INTER_LINEAR)

    def calculate_homogeneous(self):
        rotation = self.transform[:3, :3]
        translation = self.transform[:3, 3:4]
        n = rotation.I @ np.matrix([[0], [0], [-1.0]])
        d = 1.5 - translation.I @ np.matrix([[0], [0], [-1.0]])
        return self.camera_matrix @ (rotation - translation @ n.T / d) @ self.camera_matrix.I

    def warp_perspective(self, img):
        return cv2.warpPerspective(img, self.warp, self.target_resolution)

    def fine_tining(self, img, key, clear_mode=False):
        new_transform = self.fine_tining_param.update(key)
        if not clear_mode:
            self.fine_tining_param.reset()

        self.transform = new_transform if clear_mode else new_transform @ self.transform
        self.warp = self.calculate_homogeneous()
        self.rectify_x = cv2.warpPerspective(self.map[0], self.warp, self.target_resolution)
        self.rectify_y = cv2.warpPerspective(self.map[1], self.warp, self.target_resolution)
        return self(img)
