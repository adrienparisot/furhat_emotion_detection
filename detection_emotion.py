from PIL import Image
import cv2
import torch
import numpy as np
from torchvision import transforms, models
import torch.nn as nn
from furhat_remote_api import FurhatRemoteAPI
import zmq
import time
import random


#MODEL_PATH = "Dense5classes/best_finetuned_acc0.9668.pth"
MODEL_PATH = "Dense5classes/best2_finetuned_acc0.9337.pth"
CLASSES = ['angry', 'fear', 'happy', 'sad', 'surprise']

EMOTION_COOLDOWN = 3

running = True
last_emotion = None
last_emotion_time = 0
is_speaking = False
last_response = None

# Réponses pré définies
RESPONSES = {
    "angry": [
        "Je ressens de la colère. C'est une émotion forte.",
        "La colère peut être difficile à gérer parfois.",
        "Quand on est en colère, il faut essayer de respirer profondément.",
        "La colère est parfois justifiée, mais il faut savoir la canaliser.",
    ],
    "fear": [
        "J'ai peur. C'est une émotion qui nous protège du danger.",
        "La peur peut parfois nous paralyser.",
        "Respirer calmement peut aider à surmonter la peur.",
        "La peur est naturelle, elle fait partie de nous.",
    ],
    "happy": [
        "Je suis heureux ! C'est agréable de ressentir de la joie.",
        "Le bonheur est un état merveilleux à partager.",
        "Sourire est contagieux, essayez de sourire à quelqu'un aujourd'hui !",
        "La joie illumine notre journée et celle des autres.",
    ],
    "sad": [
        "Je me sens triste. C'est normal d'être triste parfois.",
        "La tristesse fait partie de la vie, comme la joie.",
        "Exprimer sa tristesse peut aider à se sentir mieux.",
        "Après la pluie vient le beau temps, la tristesse ne dure pas éternellement.",
    ],
    "surprise": [
        "Oh ! Quelle surprise !",
        "Je ne m'attendais pas à ça !",
        "Les surprises rendent la vie plus intéressante !",
        "Wow, ça c'est inattendu !",
    ],
    "neutral": [
        "Je ne perçois pas d'émotion particulière pour le moment.",
        "Tu sembles calme.",
        "C'est un moment tranquille.",
        "Je suis à l'écoute si tu veux partager quelque chose.",
        "Tout paraît normal pour l'instant.",
        "Je reste attentif à ce que tu ressens.",
        "On dirait que tu es posé et détendu.",
        "Rien de particulier à signaler, tout va bien.",
    ]
}

def get_predefined_response(emotion):
    global last_response
    choices = RESPONSES.get(emotion, ["Aucune émotion détectée."])
    response = random.choice(choices)

    while response == last_response and len(choices) > 1:
        response = random.choice(choices)

    last_response = response
    return response

# Modèle
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

# Furhat
furhat = FurhatRemoteAPI("192.168.10.14")
furhat.set_voice(name='Isabelle-Neural')
furhat.attend(user="CLOSEST")

# ZMQ caméra
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://192.168.10.14:3000")
socket.setsockopt_string(zmq.SUBSCRIBE, "")

print("[INFO] Caméra connectée")

# clean buffer
def clear_zmq_buffer():
    try:
        while True:
            socket.recv(flags=zmq.NOBLOCK)
    except:
        pass

# Expressions
def set_furhat_expression(emotion):
    expr_map = {
        "angry": {"EXPR_ANGER": 1.0, "BROW_DOWN_LEFT": 1.0, "BROW_DOWN_RIGHT": 1.0},
        "happy": {"SMILE_OPEN": 0.5, "BROW_UP_LEFT": 0.5, "BROW_UP_RIGHT": 0.5},
        "sad": {"EXPR_SAD": 1.0, "BROW_IN_LEFT": 1.0, "BROW_IN_RIGHT": 1.0},
        "fear": {"EXPR_FEAR": 1.0, "BROW_UP_LEFT": 1.0, "BROW_UP_RIGHT": 1.0},
        "surprise": {"SURPRISE": 1.0, "BROW_UP_LEFT": 1.0, "BROW_UP_RIGHT": 1.0},
        "neutre": {"SMILE_OPEN": 0.0, "SMILE_CLOSED": 0.0, "EXPR_ANGER": 0.0, "EXPR_SAD": 0.0,
                    "EXPR_FEAR": 0.0, "SURPRISE": 0.0, "PHONE_BIGAAH": 0.0, "GAZE_PAN": 0.0,
                    "GAZE_TILT": 0.0, "NECK_PAN": 0.0, "NECK_TILT": 0.0, "NECK_ROLL": 0.0}
    }

    params = expr_map.get(emotion, {})

    furhat.gesture(body={
        "name": f"{emotion}_hold",
        "class": "furhatos.gestures.Gesture",
        "frames": [
            {"time": [0.2], "params": params},
            {"time": [10.0], "params": params}
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

# Configurations led
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
    elif emotion == "neutre":
        furhat.set_led(red=0, green=0, blue=0)


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

    clear_zmq_buffer()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        tensor = transform(face).unsqueeze(0)

        with torch.no_grad():
            out = model(tensor)
            probabilities = torch.softmax(out, dim=1)
            confidence, pred = torch.max(probabilities, 1)
            if confidence < 0.6 : 
                emotion = "neutre"
            else :
                emotion = CLASSES[pred.item()]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,255), 2)

        current_time = time.time()

        if emotion and (emotion != last_emotion) and (current_time - last_emotion_time > EMOTION_COOLDOWN):

            print("[EMOTION]", emotion)
            print(confidence)
            cv2.imshow("Furhat Emotion Detection", frame)
            cv2.waitKey(1)
            
            is_speaking = True
            
            set_led(emotion)
            set_furhat_expression(emotion)

            last_emotion = emotion
            last_emotion_time = current_time

            response = get_predefined_response(emotion)
            print("[RESPONSE]", response)
            furhat.say(text=response)

            time.sleep(len(response) * 0.05)

            reset_expression()
            time.sleep(3)
            
            clear_zmq_buffer()

            is_speaking = False

    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

cv2.destroyAllWindows()
furhat.say(text="Merci pour cette interaction. À bientôt !")
