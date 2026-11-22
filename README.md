# 🐾 Woof — ResNet50 Dog Breed Classifier

An end-to-end deep learning web app that predicts a dog's breed from a photo. A fine-tuned **ResNet50** (PyTorch) is served through a **FastAPI** backend with a drag-and-drop web interface.

| Upload | Result |
|:--:|:--:|
| ![Upload screen](dogprojegorsel1.png) | ![Prediction result](dogprojegorsel2.png) |

## Features

- **Transfer learning:** ImageNet-pretrained ResNet50 with a new 10-class head.
- **Top-3 predictions** with confidence scores, shown as animated bars.
- **REST API:** `POST /predict` accepts an image and returns JSON.
- **Input validation:** JPEG / PNG / WebP / GIF only, max 10 MB, clear error messages.
- **Web UI:** drag-and-drop upload, live preview, loading state.

## Supported breeds (10)

Australian Terrier · Beagle · Border Terrier · Dingo · English Foxhound · Golden Retriever · Old English Sheepdog · Rhodesian Ridgeback · Samoyed · Shih Tzu

## Model

| | |
|---|---|
| Architecture | ResNet50 (torchvision), final `fc` layer replaced with 10 outputs |
| Pretrained on | ImageNet |
| Dataset | `[DATASET NAME / SOURCE]` |
| Train / val / test split | `[e.g. 70 / 15 / 15]` |
| Test accuracy | `[XX.X%]` |
| Optimizer / epochs | `[e.g. Adam, lr=1e-4, 15 epochs]` |
| Preprocessing | Resize 256 → CenterCrop 224 → ImageNet normalization |

## Project structure

```
.
├── main.py            # FastAPI app (routes, validation)
├── model.py           # Model loading + inference
├── classes.py         # Class names (index order = training order)
├── requirements.txt
├── templates/
│   └── index.html     # Web UI
└── resnet50_model.pth # Trained weights (download separately, see below)
```

## Getting started

```bash
git clone https://github.com/BySweK/dog-breed-classifier-fastapi.git
cd dog-breed-classifier-fastapi

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Download the weights** `resnet50_model.pth` from the [Releases](https://github.com/BySweK/dog-breed-classifier-fastapi/releases) page and place it in the project root.

```bash
uvicorn main:app --reload
```

Open http://127.0.0.1:8000 in your browser.

## API

`POST /predict` — multipart form, field name `file`.

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@my_dog.jpg"
```

```json
{
  "predictions": [
    { "label": "Beagle", "confidence": 94.21 },
    { "label": "English Foxhound", "confidence": 4.37 },
    { "label": "Border Terrier", "confidence": 0.82 }
  ]
}
```

Interactive docs are available at `/docs`.

## Notes and limitations

- The model only knows the 10 breeds above; any other image (including non-dogs) will still be assigned to one of them.
- CORS is open (`allow_origins=["*"]`) for easy local development; restrict it in production.

## Tech stack

PyTorch · Torchvision · FastAPI · Uvicorn · Pillow · HTML / Tailwind CSS / JavaScript

The web interface was built with the assistance of Cursor AI.
