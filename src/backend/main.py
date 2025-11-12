from fastapi import FastAPI
from src.backend.routes.predict import router as predict_router
from src.backend.routes.update import router as update_router

app = FastAPI(title="NBA Prediction API", version="1.0")

# Register the route groups
app.include_router(predict_router)
app.include_router(update_router)

