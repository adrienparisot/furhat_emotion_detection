import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ================= CONFIG =================
MODEL_PATH = "Dense5classes/best2_finetuned_acc0.9337.pth"
IMAGE_PATH = "photo_test/triste.png"  

CLASSES = ['angry', 'fear', 'happy', 'sad', 'surprise']

# ================= LOAD MODEL =================
print("[INFO] Chargement modèle...")

model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, len(CLASSES))

model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

print("[OK] Modèle chargé")

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ================= LOAD IMAGE =================
print("[INFO] Chargement image...")

image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# ================= INFERENCE =================
print("[INFO] Prédiction...")

with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

emotion = CLASSES[predicted.item()]

# ================= RESULT =================
print("\n===== RESULT =====")
print(f"Emotion prédite : {emotion}")
print(f"Confiance       : {confidence.item():.4f}")

print("\nDétail des probabilités :")
for i, cls in enumerate(CLASSES):
    print(f"{cls:10s} : {probabilities[0][i].item():.4f}")
