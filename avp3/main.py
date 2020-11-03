import cv2

cap = cv2.VideoCapture("test.mp4")

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# while True:
ret, frame = cap.read()
w, h, _ = frame.shape
print(w, h)

cv2.imwrite("tl.jpg", frame[:w // 2, :h // 2, :])
cv2.imwrite("tr.jpg", frame[:w // 2, h // 2:, :])
cv2.imwrite("bl.jpg", frame[w // 2:, :h // 2, :])
cv2.imwrite("br.jpg", frame[w // 2:, h // 2:, :])

cap.release()
