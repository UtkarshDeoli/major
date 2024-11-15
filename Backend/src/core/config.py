from pydantic import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "PDF Question Answering API"
    APP_DESCRIPTION: str = "API for processing PDFs and answering questions using LLM"
    APP_VERSION: str = "0.1.0"
    
    # LLM Settings
    LLM_URL: str = "http://ollama.utkarshdeoli.in/api/generate"
    LLM_MODEL: str = "llama3.2:1b"
    
    # Model settings
    EMBEDDING_MODEL: str = "paraphrase-MiniLM-L6-v2"
    MAX_TOKENS: int = 500
    CONTEXT_SIZE: int = 3  # Number of paragraphs to retrieve

    class Config:
        env_file = ".env"

settings = Settings()
