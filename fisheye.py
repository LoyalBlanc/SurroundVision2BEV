from easydict import EasyDict
import cv2
import numpy as np
from time import time

cfgs = EasyDict()
cfgs.CAMERA_ID = 6
cfgs.N_CHESS_BORAD_WIDTH = 7
cfgs.N_CHESS_BORAD_HEIGHT = 5
cfgs.CHESS_BOARD_SIZE = lambda: (cfgs.N_CHESS_BORAD_WIDTH, cfgs.N_CHESS_BORAD_HEIGHT)
cfgs.SQUARE_SIZE_MM = 20
cfgs.N_CALIBRATE_SIZE = 10
cfgs.FIND_CHESSBOARD_DELAY_MOD = 4
cfgs.FOCAL_SCALE = 1.0
cfgs.MAX_READ_FAIL_CTR = 10

flags = EasyDict()
flags.READ_FAIL_CTR = 0
flags.frame_id = 0
flags.ok = False

BOARD = np.array([[(j * cfgs.SQUARE_SIZE_MM, i * cfgs.SQUARE_SIZE_MM, 0.)]
                  for i in range(cfgs.N_CHESS_BORAD_HEIGHT) for j in range(cfgs.N_CHESS_BORAD_WIDTH)])


class CalibT(EasyDict):
    def __init__(self):
        super().__init__({
            "type": None,
            "camera_mat": None,
            "dist_coeff": None,
            "rvecs": None,
            "tvecs": None,
            "map1": None,
            "map2": None,
            "reproj_err": None,
            "ok": False,
        })


class Fisheye:
    def __init__(self):
        self.data = CalibT()
        self.inited = False

    def update(self, corners, frame_size):
        board = [BOARD] * len(corners)
        if not self.inited:
            self._update_init(board, corners, frame_size)
            self.inited = True
        else:
            self._update_refine(board, corners, frame_size)
        # self._calc_reproj_err(corners)

    def _update_init(self, board, corners, frame_size):
        data = self.data
        data.type = "FISHEYE"
        data.camera_mat = np.eye(3, 3)
        data.dist_coeff = np.zeros((4, 1))
        data.ok, data.camera_mat, data.dist_coeff, data.rvecs, data.tvecs = cv2.fisheye.calibrate(
            board, corners, frame_size, data.camera_mat, data.dist_coeff,
            flags=cv2.fisheye.CALIB_FIX_SKEW | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
            criteria=(cv2.TERM_CRITERIA_COUNT, 30, 0.1))
        data.ok = data.ok and cv2.checkRange(data.camera_mat) and cv2.checkRange(data.dist_coeff)

    def _update_refine(self, board, corners, frame_size):
        data = self.data
        data.ok, data.camera_mat, data.dist_coeff, data.rvecs, data.tvecs = cv2.fisheye.calibrate(
            board, corners, frame_size, data.camera_mat, data.dist_coeff,
            flags=cv2.fisheye.CALIB_FIX_SKEW | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.CALIB_USE_INTRINSIC_GUESS,
            criteria=(cv2.TERM_CRITERIA_COUNT, 10, 0.1))
        data.ok = data.ok and cv2.checkRange(data.camera_mat) and cv2.checkRange(data.dist_coeff)

    def _calc_reproj_err(self, corners):
        if not self.inited:
            return
        data = self.data
        data.reproj_err = []
        for i in range(len(corners)):
            corners_reproj = cv2.fisheye.projectPoints(BOARD[i], data.rvecs[i], data.tvecs[i], data.camera_mat,
                                                       data.dist_coeff)
            err = cv2.norm(corners_reproj, corners[i], cv2.NORM_L2)
            data.reproj_err.append(err)


class DataT(EasyDict):
    def __init__(self, temp_frame):
        super().__init__({
            "raw_frame": temp_frame,
            "corners": None,
            "ok": False,
        })
        # find chess board
        self.ok, self.corners = cv2.findChessboardCorners(
            self.raw_frame, cfgs.CHESS_BOARD_SIZE(),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)
        if not self.ok:
            return
        # subpix
        gray = cv2.cvtColor(self.raw_frame, cv2.COLOR_BGR2GRAY)
        self.corners = cv2.cornerSubPix(gray, self.corners, (11, 11), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1))


class HistoryT:
    def __init__(self):
        self.corners = []
        self.updated = False

    def append(self, current_frame):
        if not current_frame.ok:
            return
        self.corners.append(current_frame.corners)
        self.updated = True

    def removei(self, i):
        if not 0 <= i < len(self):
            return
        del self.corners[i]
        self.updated = True

    def __len__(self):
        return len(self.corners)

    def get_corners(self):
        self.updated = False
        return self.corners


history = HistoryT()

cap = cv2.VideoCapture(cfgs.CAMERA_ID)
if not cap.isOpened():
    raise RuntimeError("camera open failed")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

fisheye = Fisheye()

while True:
    ok, raw_frame = cap.read()
    if not ok:
        flags.READ_FAIL_CTR += 1
        if flags.READ_FAIL_CTR >= cfgs.MAX_READ_FAIL_CTR:
            raise RuntimeError("image read failed")
    else:
        flags.READ_FAIL_CTR = 0
        flags.frame_id += 1

    if 0 == flags.frame_id % cfgs.FIND_CHESSBOARD_DELAY_MOD:
        current = DataT(raw_frame)
        history.append(current)

    calib = None
    if len(history) >= cfgs.N_CALIBRATE_SIZE and history.updated:
        fisheye.update(history.get_corners(), raw_frame.shape[1::-1])
        calib = fisheye.data
        calib.map1, calib.map2 = cv2.fisheye.initUndistortRectifyMap(
            calib.camera_mat, calib.dist_coeff, np.eye(3, 3), calib.camera_mat, raw_frame.shape[1::-1], cv2.CV_16SC2)

    if len(history) >= cfgs.N_CALIBRATE_SIZE:
        undist_frame = cv2.remap(raw_frame, calib.map1, calib.map2, cv2.INTER_LINEAR)
        resize0 = cv2.resize(undist_frame, (640, 480))
        cv2.imshow("undist_frame", resize0)
    resize1 = cv2.resize(raw_frame, (640, 480))
    cv2.imshow("raw_frame", resize1)
    key = cv2.waitKey(1)
    if key == 27:
        break

with open('undistort.txt', 'a+') as f:
    script = f"""
        {time()}
        "ID": {cfgs.CAMERA_ID}    
        "K": np.matrix({fisheye.data.camera_mat.tolist()}),
        "D": np.matrix({fisheye.data.dist_coeff.tolist()}),
        
    """
    f.write(str(script))
