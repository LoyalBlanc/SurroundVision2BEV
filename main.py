import cv2

from camera import Camera
from camera_param import *

if __name__ == "__main__":
    cam_back = Camera(BACK, ORIGINAL_RESOLUTION, TARGET_RESOLUTION)
    cam_front = Camera(FRONT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION)
    cam_left = Camera(LEFT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION)
    cam_right = Camera(RIGHT, ORIGINAL_RESOLUTION, TARGET_RESOLUTION)

    cap = cv2.VideoCapture("input/test.mp4")
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    video = cv2.VideoWriter("output/test.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 5, (1000, 1000))

    index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            index += 1

        if index % 25 == 0:
            print(index)
            img_back = cam_back(frame[:ORIGINAL_RESOLUTION[0] // 2, ORIGINAL_RESOLUTION[1] // 2:, :])
            img_front = cam_front(frame[:ORIGINAL_RESOLUTION[0] // 2, :ORIGINAL_RESOLUTION[1] // 2, :])
            img_left = cam_left(frame[ORIGINAL_RESOLUTION[0] // 2:, :ORIGINAL_RESOLUTION[1] // 2, :])
            img_right = cam_right(frame[ORIGINAL_RESOLUTION[0] // 2:, ORIGINAL_RESOLUTION[1] // 2:, :])

            bf = cv2.bitwise_or(img_back, img_front)
            lr = cv2.bitwise_or(img_left, img_right)
            image = (np.where(lr, lr, bf) + np.where(bf, bf, lr)) // 2

            video.write(image)

    cap.release()
    video.release()
