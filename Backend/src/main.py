from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from src.services.bm25_index import BM25IndexService
from src.core.data_store import get_user_chunks_for_bm25, ensure_indexes
from src.core.security import get_current_user

# Import our routers
from src.routers import auth_router, pdf_router, document_router, question_router, analysis_router, mock_test_router
from src.routers.exam_router import router as exam_router
from src.routers.subject_router import router as subject_router
from src.routers.collection_router import router as collection_router
from src.routers.material_router import router as material_router
from src.routers.onboarding_router import router as onboarding_router

app = FastAPI()

# Allow CORS for frontend origin with credentials
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include our routers
app.include_router(auth_router.router)
app.include_router(pdf_router.router)
app.include_router(question_router.router)
app.include_router(analysis_router.router)
app.include_router(mock_test_router.router)
app.include_router(document_router)
app.include_router(exam_router)
app.include_router(subject_router)
app.include_router(collection_router)
app.include_router(material_router)
app.include_router(onboarding_router)

bm25_service = BM25IndexService()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    try:
        # Ensure MongoDB indexes exist
        await ensure_indexes()
        
        # Build BM25 indexes for all users
        # Note: For production with many users, consider lazy loading
        # For now, we build on first query to avoid startup delays
        pass
    except Exception as e:
        print(f"Startup initialization warning: {e}")

@app.post("/admin/rebuild-index")
async def rebuild_index(user_id: str = Depends(get_current_user)):
    """Rebuild BM25 index for the current user."""
    chunks = await get_user_chunks_for_bm25(user_id)
    bm25_service.build_index(user_id, chunks)
    return {"status": "ok", "chunk_count": len(chunks)}

@app.get("/")
async def root():
    return {"message": "Welcome to Orbit API"}

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)