from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import nltk
from typing import Dict

# Import our routers
from src.routers import auth_router, pdf_router, question_router

app = FastAPI()

# Allow CORS for all origins (you can restrict this to specific origins if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include our routers
app.include_router(auth_router.router)
app.include_router(pdf_router.router)
app.include_router(question_router.router)

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