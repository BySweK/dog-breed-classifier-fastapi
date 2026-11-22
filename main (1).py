from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from model import predict

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Woof — Dog Breed Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # development only; restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_SIZE_MB = 10


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.post("/predict")
async def predict_breed(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen dosya tipi: {file.content_type}. "
                   "Lütfen JPEG, PNG veya WebP yükleyin.",
        )

    image_bytes = await file.read()

    if len(image_bytes) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"Dosya boyutu {MAX_SIZE_MB} MB'ı geçemez.",
        )

    try:
        results = predict(image_bytes, top_k=3)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin hatası: {str(e)}")

    return {"predictions": results}
