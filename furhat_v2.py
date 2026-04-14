from furhat_remote_api import FurhatRemoteAPI
import json
import requests
import zmq
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms



context = zmq.Context()

socket = context.socket(zmq.SUB)

socket.connect("tcp://192.168.10.14:3000")  # 🔥 IP Furhat
socket.setsockopt_string(zmq.SUBSCRIBE, "")



MODEL_PATH = "Dense5classes/best2_finetuned_acc0.9587.pth"
CLASSES = ['angry', 'fear', 'happy', 'sad', 'surprise']

#MODEL_PATH = "emotion_detection_models/fine_tuned_model.pth"
#CLASSES = ['angry', 'fear', 'happy', 'sad']

# ================= LOAD MODEL =================
print("[INFO] Chargement modèle...")

emotion_model = models.densenet121(weights=None)
emotion_model.classifier = nn.Linear(
    emotion_model.classifier.in_features,
    len(CLASSES)
)

emotion_model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
emotion_model.eval()

print("[OK] Modèle chargé")

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])                         

# NOTE: ollama must be running for this to work, start the ollama app or run `ollama serve`
model = "mixtral"  # TODO: update this for whatever model you wish to use
import time

mots_cles_reset=["reset", "clear", "remplace"]
mots_cles_terminer=["terminé", "terminer", "exit"]


prompts_files = {
    "prose": "prompts/prompt_3.txt",
    "prompt_0": "prompts/prompt_0.txt",
    "prompt_1": "prompts/prompt_1.txt"
}

current_prompt = "prompt_1"


"""
def get_frame():
    try:
        msg = socket.recv(flags=zmq.NOBLOCK)
        img = np.frombuffer(msg, dtype=np.uint8)
        frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
        return frame
    except:
        return None
"""

        
def get_frame():
    frame = None

    while True:
        try:
            msg = socket.recv(flags=zmq.NOBLOCK)

            # ignore petits messages (heartbeat / metadata)
            if len(msg) < 5000:
                continue

            img = np.frombuffer(msg, dtype=np.uint8)
            decoded = cv2.imdecode(img, cv2.IMREAD_COLOR)

            if decoded is not None:
                frame = decoded  # on écrase → garde le dernier frame

        except:
            break  # buffer complètement vidé

    if frame is None:
        print("[DEBUG] Aucun frame valide reçu")
    else:
        print("[DEBUG] Frame OK shape:", frame.shape)

    return frame

"""
def detect_emotion_once_zmq():
    print("[EMOTION] Capture Furhat...")

    frame = get_frame()
    print("aaaaa")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "no_face"

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        tensor = transform(face).unsqueeze(0)

        with torch.no_grad():
            out = emotion_model(tensor)
            _, pred = torch.max(out, 1)

        return CLASSES[pred.item()]

    return "unknown"
"""
"""
def detect_emotion_once_zmq():
    print("[EMOTION] Capture Furhat...")
    
    frame = None
    timeout = time.time() + 2  # 2 sec max

    while frame is None and time.time() < timeout:
        frame = get_frame()

    if frame is None:
        return "no_frame"

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "no_face"

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        tensor = transform(face).unsqueeze(0)

        with torch.no_grad():
            out = emotion_model(tensor)
            _, pred = torch.max(out, 1)

        return CLASSES[pred.item()]

    return "unknown"
 """
 
"""
def detect_emotion_once_zmq():
    print("[EMOTION] Attente visage Furhat...")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    timeout = time.time() + 5  # ⏳ max 5 sec

    frame = None
    faces = []

    # ===============================
    # 1. attendre un frame valide
    # ===============================
    while time.time() < timeout:

        frame = get_frame()
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            break

    # ===============================
    # 2. si toujours rien → retry message
    # ===============================
    if frame is None or len(faces) == 0:
        print("[EMOTION] aucun visage détecté")
        return "no_face"

    # ===============================
    # 3. prediction sur 1er visage
    # ===============================
    (x, y, w, h) = faces[0]

    face = frame[y:y+h, x:x+w]
    face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    tensor = transform(face).unsqueeze(0)

    with torch.no_grad():
        out = emotion_model(tensor)
        _, pred = torch.max(out, 1)

    emotion = CLASSES[pred.item()]
    return emotion 
"""

# ================= EXPRESSION CONTINUE =================
def set_furhat_expression(furhat, emotion):
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

def reset_expression(furhat):
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
def set_led(furhat, emotion):
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

def detect_emotion_once_zmq():
    print("[EMOTION] Attente visage Furhat...")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    timeout = time.time() + 5  # max 5 sec

    frame = None
    faces = []

    # ===============================
    # 1. attendre un frame valide
    # ===============================
    while time.time() < timeout:

        frame = get_frame()
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            break

    # ===============================
    # 2. si rien détecté
    # ===============================
    if frame is None or len(faces) == 0:
        print("[EMOTION] aucun visage détecté")
        return "no_face"

    # ===============================
    # 3. draw + prediction
    # ===============================
    (x, y, w, h) = faces[0]

    # draw rectangle sur frame
    debug_frame = frame.copy()
    cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    face = frame[y:y+h, x:x+w]
    face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    tensor = transform(face).unsqueeze(0)

    with torch.no_grad():
        out = emotion_model(tensor)
        probabilities = torch.softmax(out, dim=1)
        _, pred = torch.max(probabilities, 1)

    emotion = CLASSES[pred.item()]
    


    # afficher texte sur image
    cv2.putText(
        debug_frame,
        emotion,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # ===============================
    # 4. DISPLAY FRAME
    # ===============================
    cv2.imshow("Emotion Detection Debug", debug_frame)
    cv2.waitKey(1)

    return emotion
 
def chat(messages):
    r = requests.post(
        "http://127.0.0.1:11434/api/chat",       # IP ollama server
        #"http://130.79.207.171:11434/api/chat",
        json={"model": model, "messages": messages, "stream": True},
    )
    r.raise_for_status()
    output = ""
    print("[Furhat] ")
    for line in r.iter_lines():
        body = json.loads(line)
        if "error" in body:
            raise Exception(body["error"])
        if body.get("done") is False:
            message = body.get("message", "")
            content = message.get("content", "")
            output += content
            # the response streams one token at a time, print that as we receive it
            print(content, end="", flush=True)

        if body.get("done", False):
            message["content"] = output
            return message


def test_change_prompt(com, messages):
    if com in prompts_files:

        print(f"Chargement du prompt : {com}")

        current_prompt = com

        with open(prompts_files[current_prompt], "r") as f:
            prompt = f.read()
        
        return [{"role": "user", "content": prompt}]
    return messages



def main():
    messages = []
    
    with open(prompts_files[current_prompt], "r") as f:
        prompt = f.read()
    
    messages.append({"role": "user", "content": prompt})
    message = chat(messages)
    messages.append(message)       
    print("\n\n")
    
    # Create an instance of the FurhatRemoteAPI class, providing the address of the robot or the SDK running the virtual robot
    #furhat = FurhatRemoteAPI("192.168.1.180")
    furhat = FurhatRemoteAPI("192.168.10.14")     # IP Furhat, robot should be in "Remote API mode"
    #furhat.setInputLanguage(Language.FRENCH_FR)

    # Set the LED lights
    furhat.set_led(red=80, green=80, blue=200)

    # Get the voices on the robot
    voices = furhat.get_voices()
    print(voices)
    # Set the voice of the robot
    furhat.set_voice(name='Isabelle-Neural')
    
    # Get the users detected by the robot 
    users = furhat.get_users()
    print(users)
    # Attend the user closest to the robot
    furhat.attend(user="CLOSEST")

    user_input = ''
    # Say "Hi there!"
    furhat.say(text="bonjour!")
    while user_input != 'terminé'.strip() and message != 'terminer'.strip() :

        print("Appuyer sur entrée et parler\n")
        com = input(">>>")
        if com in mots_cles_reset:
            with open(prompts_files[current_prompt], "r") as f:
                prompt = f.read()
            messages = [{"role": "user", "content": prompt}]

        if com in mots_cles_terminer:
            return

        messages = test_change_prompt(com, messages)

        while user_input == '':
            result = furhat.listen(language='fr-FR')
            user_input = result.message
            print('--- attente ---')
            furhat.attend(user="RANDOM")

        user_input = user_input.replace("immeuble", "InnovLab")

        print('[Vous] ', user_input)
        
        if "emotion" in user_input.lower() or "émotion" in user_input.lower():
            #reset_expression(furhat)
            emotion = detect_emotion_once_zmq()
            print("[RESULT EMOTION]", emotion)
            set_led(furhat, emotion)
            set_furhat_expression(furhat, emotion)
            response = ask_ollama(emotion)
            print("[OLLAMA]", response)

            furhat.say(text=response)
            time.sleep(max(3, len(response) * 0.05))
            #furhat.say(text=f"Je pense que ton émotion est {emotion}")
            user_input = ''   # 🔥 IMPORTANT
            reset_expression(furhat)
            continue

        #furhat.attend(location="0.0,0.2,1.0")
        furhat.attend(user="RANDOM")
        #print(type(message))
        messages.append({"role": "user", "content": user_input})
        message = chat(messages)
        furhat.say(text = message["content"], blocking=True)	
        messages.append(message)
        user_input = ''
        print("\n\n")
        
    furhat.say(text="au revoir, et à bientôt!")

if __name__ == "__main__":
    main()









