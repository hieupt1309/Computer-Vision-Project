import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"Camera {i}: OK  (frame: {frame.shape})")
        else:
            print(f"Camera {i}: opened but no frame")
        cap.release()
    else:
        print(f"Camera {i}: cannot open")
