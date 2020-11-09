import cv2
import numpy as np


class Camera(object):
    def __init__(self, camera_param, original_resolution, target_resolution):
        self.camera_matrix = camera_param['K']
        self.dist_coefficient = camera_param['D']
        self.rotation = camera_param['R']
        self.translation = camera_param['T']

        self.original_resolution = original_resolution
        self.target_resolution = target_resolution
        self.camera_target = np.array([[100., 0., original_resolution[0] / 2.],
                                       [0., 100., original_resolution[1] / 2.],
                                       [0., 0., 1.]])

        self.map = self.calculate_rectify(
            self.camera_matrix, self.dist_coefficient, self.camera_target, self.original_resolution)
        self.warp = self.calculate_homogeneous(
            self.camera_matrix, self.rotation, self.translation, np.matrix([[0], [0], [-1]]))

    @staticmethod
    def calculate_rectify(camera_matrix, dist_coefficient, camera_target, original_resolution):
        return cv2.fisheye.initUndistortRectifyMap(
            camera_matrix, dist_coefficient, np.eye(3),
            camera_target, original_resolution, cv2.CV_32F)

    @staticmethod
    def calculate_homogeneous(camera_matrix, rotation, translation, n):
        return camera_matrix * (rotation + translation * n.T) * camera_matrix.I

    def __call__(self, img):
        # img = cv2.remap(img, self.map[0], self.map[1], cv2.INTER_LINEAR)
        img = cv2.warpPerspective(img, self.warp, self.target_resolution)
        return img

    def fine_tine(self, x=0., y=0., z=0., roll=0., pitch=0., yaw=0.):
        """ approximate new RT """
        roll_rotation = np.matrix([[1, 0, 0],
                                   [0, np.cos(roll), -np.sin(roll)],
                                   [0, np.sin(roll), np.cos(roll)]])

        pitch_rotation = np.matrix([[np.cos(pitch), 0, -np.sin(pitch)],
                                    [0, 1, 0],
                                    [np.sin(pitch), 0, np.cos(pitch)]])

        yaw_rotation = np.matrix([[np.cos(yaw), -np.sin(yaw), 0],
                                  [np.sin(yaw), np.cos(yaw), 0],
                                  [0, 0, 1]])

        # rotation = (roll_rotation * pitch_rotation * yaw_rotation)
        # translation = np.matrix([[- x], [- y], [- z]])
        # warp_n = rotation.I @ np.matrix([[0], [0], [-1]])
        # self.warp = self.calculate_homogeneous(self.camera_matrix, self.rotation, self.translation, warp_n)
        # self.fine = self.calculate_homogeneous(self.camera_matrix, rotation, translation, np.matrix([[0], [0], [-1]]))

        self.rotation = (roll_rotation * pitch_rotation * yaw_rotation) @ self.rotation
        self.translation = np.matrix([[self.translation[0].item() - x],
                                      [self.translation[1].item() - y],
                                      [self.translation[2].item() - z]])
        warp_n = np.matrix([[0], [0], [-1]])
        self.warp = self.calculate_homogeneous(self.camera_matrix, self.rotation, self.translation, warp_n)
