from PIL import Image
import cv2
import torch
import numpy as np
from torchvision import transforms, models
import torch.nn as nn
import requests
from furhat_remote_api import FurhatRemoteAPI
import zmq
import time

# ================= CONFIG =================
MODEL_PATH = "Dense5classes/best2_finetuned_acc0.9587.pth"
CLASSES = ['angry', 'fear', 'happy', 'sad', 'surprise']

EMOTION_COOLDOWN = 3

running = True
last_emotion = None
last_emotion_time = 0
is_speaking = False

# ================= MODEL =================
print("[INFO] Chargement modèle...")
model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()
print("[OK] Modèle chargé")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ================= FURHAT =================
furhat = FurhatRemoteAPI("192.168.10.14")
furhat.set_voice(name='Isabelle-Neural')
furhat.attend(user="CLOSEST")

# ================= ZMQ =================
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://192.168.10.14:3000")
socket.setsockopt_string(zmq.SUBSCRIBE, "")

print("[INFO] Caméra connectée")

# ================= BUFFER =================
def clear_zmq_buffer():
    try:
        while True:
            socket.recv(flags=zmq.NOBLOCK)
    except:
        pass

# ================= EXPRESSION CONTINUE =================
def set_furhat_expression(emotion):
    expr_map = {
        "angry": {"EXPR_ANGER": 1.0, "BROW_DOWN_LEFT": 1.0, "BROW_DOWN_RIGHT": 1.0, "SMILE_OPEN": 0.0, "PHONE_BIGAAH":0.0},
        "happy": {"BLINK_LEFT": 1.0, "BROW_UP_LEFT": 0.5, "BROW_UP_RIGHT": 0.5, "SMILE_OPEN": 0.4, "PHONE_BIGAAH":0.0},
        "sad": {"EXPR_SAD": 1.0, "BROW_IN_LEFT": 1.0, "BROW_IN_RIGHT": 1.0, "SMILE_OPEN": 0.1, "PHONE_BIGAAH":0.0},
        "fear": {"EXPR_FEAR": 1.0, "BROW_UP_LEFT": 1.0, "BROW_UP_RIGHT": 1.0, "SMILE_OPEN": 0.1, "PHONE_BIGAAH":0.0},
        "surprise": {"SURPRISE": 1.0, "BROW_UP_LEFT": 1.0, "BROW_UP_RIGHT": 1.0, "PHONE_BIGAAH":1.0}
    }

    params = expr_map.get(emotion, {})

    # 👉 expression tenue longtemps
    furhat.gesture(body={
        "name": f"{emotion}_hold",
        "class": "furhatos.gestures.Gesture",
        "frames": [
            {
                "time": [0.2],
                "params": params
            },
            {
                "time": [10.0],  # 🔥 longue durée (sera coupée après)
                "params": params
            }
        ]
    })

def reset_expression():
    furhat.gesture(body={
        "name": "reset_full",
        "class": "furhatos.gestures.Gesture",
        "frames": [
            {
                "time": [0.3],
                "params": {
                    "reset": True,
                    "SMILE_OPEN": 0.0,
                    "SMILE_CLOSED": 0.0,
                    "EXPR_ANGER": 0.0,
                    "EXPR_SAD": 0.0,
                    "EXPR_FEAR": 0.0,
                    "SURPRISE": 0.0,
                    "PHONE_BIGAAH": 0.0,
                    "GAZE_PAN": 0.0,
                    "GAZE_TILT": 0.0,
                    "NECK_PAN": 0.0,
                    "NECK_TILT": 0.0,
                    "NECK_ROLL": 0.0
                }
            },
            {
                "time": [1.0],
                "params": {"reset": True}
            }
        ]
    })

# ================= LED =================
def set_led(emotion):
    if emotion == "angry":
        furhat.set_led(red=255, green=0, blue=0)
    elif emotion == "happy":
        furhat.set_led(red=255, green=223, blue=0)
    elif emotion == "sad":
        furhat.set_led(red=80, green=130, blue=180)
    elif emotion == "fear":
        furhat.set_led(red=120, green=0, blue=120)
    elif emotion == "surprise":
        furhat.set_led(red=0, green=255, blue=0)

# ================= OLLAMA =================
def ask_ollama(emotion):
    prompt = f"L'utilisateur semble {emotion}. Réponds en français uniquement en 2 phrases maximum avec un ton empathique."

    try:
        r = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "mixtral:latest",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            },
            timeout=30
        )
        return r.json()["message"]["content"]
    except:
        return "Je ne peux pas répondre."

# ================= LOOP =================
print("[INFO] Démarrage...")

while running:
    try:
        msg = socket.recv()
        img = np.frombuffer(msg, dtype=np.uint8)
        frame = cv2.imdecode(img, cv2.IMREAD_COLOR)

        if frame is None:
            continue

    except Exception as e:
        print("[ERREUR caméra]", e)
        continue

    if is_speaking:
        cv2.imshow("Furhat", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
        continue

    # 🔥 flush buffer
    clear_zmq_buffer()

    # ================= DETECTION =================
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    emotion = None

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        tensor = transform(face).unsqueeze(0)

        with torch.no_grad():
            out = model(tensor)
            _, pred = torch.max(out, 1)
            emotion = CLASSES[pred.item()]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,255), 2)

        current_time = time.time()

        if emotion and (emotion != last_emotion) and (current_time - last_emotion_time > EMOTION_COOLDOWN):

            print("[EMOTION]", emotion)

            is_speaking = True

            set_led(emotion)
            set_furhat_expression(emotion)

            last_emotion = emotion
            last_emotion_time = current_time

            response = ask_ollama(emotion)
            print("[OLLAMA]", response)

            furhat.say(text=response)

            # ⏳ attendre fin parole
            time.sleep(max(3, len(response) * 0.05))

            # 🔥 reset après parole
            reset_expression()
            clear_zmq_buffer()

            is_speaking = False

    cv2.imshow("Furhat Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

cv2.destroyAllWindows()
furhat.say(text="Merci pour cette interaction. À bientôt !")
