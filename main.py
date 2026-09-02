from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

from model import predict

app = FastAPI(title="Woof — Dog Breed Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_SIZE_MB = 10


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict_breed(file: UploadFile = File(...)):
    # Dosya tipi kontrolü
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen dosya tipi: {file.content_type}. "
                   "Lütfen JPEG, PNG veya WebP yükleyin.",
        )

    image_bytes = await file.read()

    # Boyut kontrolü
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
