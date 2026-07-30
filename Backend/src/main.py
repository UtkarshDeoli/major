from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import nltk

from src.core.config import FRONTEND_URL, RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET
from src.core.limiter import limiter
from src.core.data_store import client as mongo_client

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
    class_students_router,
    class_subject_router,
    class_material_router,
    sample_material_router,
    subscription_router,
    webhook_router,
    org_router,
    admin_router,
    socratic_router,
    study_router,
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

# Rate limiting (slowapi)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
app.include_router(class_students_router)
app.include_router(class_subject_router)
app.include_router(class_material_router)
app.include_router(sample_material_router)
app.include_router(subscription_router)
app.include_router(webhook_router)
app.include_router(org_router)
app.include_router(admin_router)
app.include_router(socratic_router)
app.include_router(study_router)

@app.get("/")
async def root():
    return {"message": "Welcome to Padhai Whallah API"}

async def _check_mongodb() -> dict:
    """Best-effort MongoDB connectivity check."""
    if mongo_client is None:
        return {"ok": False, "error": "client not initialized"}
    try:
        await mongo_client.admin.command("ping")
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def _check_razorpay() -> dict:
    """Best-effort Razorpay readiness check (non-fatal)."""
    if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
        return {"ok": False, "error": "keys not configured"}
    try:
        import razorpay
        client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
        # Lightweight, low-risk API call to verify credentials.
        client.plan.all()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/healthcheck")
async def healthcheck():
    mongodb = await _check_mongodb()
    razorpay = await _check_razorpay()
    status_code = 200 if mongodb["ok"] else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ok" if mongodb["ok"] else "degraded",
            "mongodb": mongodb,
            "razorpay": razorpay,
        },
    )

if __name__ == "__main__":
    import uvicorn
    # Download necessary NLTK data
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    uvicorn.run(app, host="0.0.0.0", port=8001)