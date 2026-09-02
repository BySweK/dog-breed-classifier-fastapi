import io
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

from classes import DOG_CLASSES

# ── Dönüşüm pipeline'ı (ImageNet normalizasyonu) ─────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

NUM_CLASSES = len(DOG_CLASSES)  # 10

# ── Model yükleme ─────────────────────────────────────────────────────────────
def _load_model(weights_path: str = "resnet50_model.pth") -> torch.nn.Module:
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

    state = torch.load(weights_path, map_location="cpu")

    # Tam model kaydedilmişse state_dict'i çek
    if isinstance(state, torch.nn.Module):
        state = state.state_dict()
    # 'model_state_dict' anahtarıyla kaydedilmişse
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)
    model.eval()
    return model


_model: torch.nn.Module = _load_model()


# ── Tahmin fonksiyonu ─────────────────────────────────────────────────────────
def predict(image_bytes: bytes, top_k: int = 3) -> list[dict]:

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0)  # (1, 3, 224, 224)

    with torch.no_grad():
        logits = _model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    top_probs, top_indices = torch.topk(probs, k=min(top_k, NUM_CLASSES))

    results = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        results.append({
            "label": DOG_CLASSES[idx],
            "confidence": round(prob * 100, 2),
        })
    return results
