import cv2
import numpy as np


class Camera(object):
    def __init__(self, camera_param, original_resolution, target_resolution):
        self.camera_matrix = camera_param['K']
        self.dist_coefficient = camera_param['D']
        self.transform = camera_param['T']
        self.original_resolution = original_resolution
        self.target_resolution = target_resolution

        self.map = self.calculate_rectify()
        self.warp = self.calculate_homogeneous()

    def calculate_rectify(self):
        # camera_target = np.matrix([[50., 0., self.original_resolution[0] / 2.],
        #                            [0., 50., self.original_resolution[1] / 2.],
        #                            [0., 0., 1.]])
        return cv2.fisheye.initUndistortRectifyMap(self.camera_matrix, self.dist_coefficient, np.eye(3),
                                                   self.camera_matrix, self.original_resolution, cv2.CV_16SC2)

    def calculate_homogeneous(self):
        rotation = self.transform[:3, :3]
        translation = self.transform[:3, 3:4]
        n = rotation.I @ np.matrix([[0], [0], [-1.0]])
        d = 1.5 - translation.I @ np.matrix([[0], [0], [-1.0]])
        return self.camera_matrix @ (rotation - translation @ n.T / d) @ self.camera_matrix.I

    def __call__(self, img):
        img = cv2.remap(img, self.map[0], self.map[1], cv2.INTER_LINEAR)
        # img = cv2.warpPerspective(img, self.warp, self.target_resolution)
        return img

    def fine_tine(self, x=0., y=0., z=0., roll=0., pitch=0., yaw=0., clear_mode=True):
        roll_rotation = np.matrix([[1, 0, 0],
                                   [0, np.cos(roll), -np.sin(roll)],
                                   [0, np.sin(roll), np.cos(roll)]])
        pitch_rotation = np.matrix([[np.cos(pitch), 0, -np.sin(pitch)],
                                    [0, 1, 0],
                                    [np.sin(pitch), 0, np.cos(pitch)]])
        yaw_rotation = np.matrix([[np.cos(yaw), -np.sin(yaw), 0],
                                  [np.sin(yaw), np.cos(yaw), 0],
                                  [0, 0, 1]])

        rotation = (roll_rotation * pitch_rotation * yaw_rotation)
        translation = np.matrix([[x], [y], [z]])
        new_transform = np.concatenate((np.concatenate((rotation, translation), axis=1),
                                        np.matrix([0, 0, 0, 1])), axis=0)

        self.transform = new_transform if clear_mode else new_transform @ self.transform
        self.warp = self.calculate_homogeneous()
