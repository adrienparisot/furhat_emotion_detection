import zmq
import cv2
import numpy as np
import json
import os

# CONFIG
ZMQ_URL = "tcp://192.168.10.14:3000"  
SAVE_DIR = "dataset"
EMOTION_LABEL = input("Emotion (happy, sad, etc) : ")

save_path = os.path.join(SAVE_DIR, EMOTION_LABEL)
os.makedirs(save_path, exist_ok=True)

# face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ZMQ
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect(ZMQ_URL)
socket.setsockopt_string(zmq.SUBSCRIBE, "")

img_count = 1000

print("'s' = save face | 'q' = quit")

while True:
    jpg_bytes = socket.recv()
    meta_bytes = socket.recv()

    img = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # draw boxes
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x+3, y+3), (x + w+3, y + h+3), (0, 255, 0), 2)

    cv2.imshow("Furhat Camera", img)

    key = cv2.waitKey(1) & 0xFF

    # SAVE FACE ONLY
    if key == ord('s'):
        if len(faces) == 0:
            print("Aucun visage détecté")
            continue

        # prendre le premier visage
        x, y, w, h = faces[0]
        face = img[y+5:y+h-3, x+5:x+w-3]

        filename = os.path.join(save_path, f"{EMOTION_LABEL}_{img_count}.jpg")
        cv2.imwrite(filename, face)

        print(f"Face saved: {filename}")
        img_count += 1

    elif key == ord('q'):
        break

cv2.destroyAllWindows()
