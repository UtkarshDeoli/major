from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import nltk

from src.core.config import FRONTEND_URL

# Import our routers
from src.routers import (
    auth_router,
    pdf_router,
    document_router,
    question_router,
    analysis_router,
    mock_test_router,
    teacher_router,
    analytics_router,
    exam_router,
    subject_router,
    collection_router,
    material_router,
    onboarding_router,
    flashcard_router,
    ai_material_router,
    class_router,
    sample_material_router,
    subscription_router,
    webhook_router,
    org_router,
    admin_router,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Ensure MongoDB indexes exist on startup (best-effort)."""
    from src.core.data_store import ensure_indexes
    try:
        await ensure_indexes()
    except Exception as e:  # pragma: no cover - best-effort index creation
        print(f"Startup ensure_indexes failed: {e}")
    yield


app = FastAPI(lifespan=lifespan)

# Allow CORS for the frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include our routers
app.include_router(auth_router)
app.include_router(pdf_router)
app.include_router(document_router)
app.include_router(question_router)
app.include_router(analysis_router)
app.include_router(mock_test_router)
app.include_router(teacher_router)
app.include_router(analytics_router)
app.include_router(exam_router)
app.include_router(subject_router)
app.include_router(collection_router)
app.include_router(material_router)
app.include_router(onboarding_router)
app.include_router(flashcard_router)
app.include_router(ai_material_router)
app.include_router(class_router)
app.include_router(sample_material_router)
app.include_router(subscription_router)
app.include_router(webhook_router)
app.include_router(org_router)
app.include_router(admin_router)

@app.get("/")
async def root():
    return {"message": "Welcome to Padhai Whallah API"}

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    # Download necessary NLTK data
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    uvicorn.run(app, host="0.0.0.0", port=8001)