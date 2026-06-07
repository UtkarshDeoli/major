from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import our routers
from src.routers import auth_router, pdf_router, question_router, analysis_router, mock_test_router
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
app.include_router(exam_router)
app.include_router(subject_router)
app.include_router(collection_router)
app.include_router(material_router)
app.include_router(onboarding_router)

@app.get("/")
async def root():
    return {"message": "Welcome to Padhai Whallah API"}

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)