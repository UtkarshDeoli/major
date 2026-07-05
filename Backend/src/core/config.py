import os
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # MongoDB settings
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "phadai"
    MONGODB_CONNECT_TIMEOUT: int = 30000  # 30 seconds timeout

    # JWT settings
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # ChromaDB settings
    CHROMA_DB_PATH: str = "./chroma_db"
    GEMINI_API_KEY: str

    # Google OAuth settings
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = "http://localhost:8001/auth/google/callback"
    FRONTEND_URL: str = "http://localhost:3000"

    # Razorpay (India-first billing)
    RAZORPAY_KEY_ID: str = ""
    RAZORPAY_KEY_SECRET: str = ""
    RAZORPAY_WEBHOOK_SECRET: str = ""
    RAZORPAY_PLAN_PRO_MONTHLY: str = ""
    RAZORPAY_PLAN_PREMIUM_MONTHLY: str = ""
    RAZORPAY_PLAN_PRO_YEARLY: str = ""
    RAZORPAY_PLAN_PREMIUM_YEARLY: str = ""
    RZP_TEST_MODE: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

# Create settings instance
settings = Settings()

# Export settings variables
MONGODB_URL = settings.MONGODB_URL
MONGODB_DB_NAME = settings.MONGODB_DB_NAME
MONGODB_CONNECT_TIMEOUT = settings.MONGODB_CONNECT_TIMEOUT
SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES
CHROMA_DB_PATH = settings.CHROMA_DB_PATH
GEMINI_API_KEY = settings.GEMINI_API_KEY
GOOGLE_CLIENT_ID = settings.GOOGLE_CLIENT_ID
GOOGLE_CLIENT_SECRET = settings.GOOGLE_CLIENT_SECRET
GOOGLE_REDIRECT_URI = settings.GOOGLE_REDIRECT_URI
FRONTEND_URL = settings.FRONTEND_URL

# Razorpay (India-first billing)
RAZORPAY_KEY_ID = settings.RAZORPAY_KEY_ID
RAZORPAY_KEY_SECRET = settings.RAZORPAY_KEY_SECRET
RAZORPAY_WEBHOOK_SECRET = settings.RAZORPAY_WEBHOOK_SECRET
RAZORPAY_PLAN_PRO_MONTHLY = settings.RAZORPAY_PLAN_PRO_MONTHLY
RAZORPAY_PLAN_PREMIUM_MONTHLY = settings.RAZORPAY_PLAN_PREMIUM_MONTHLY
RAZORPAY_PLAN_PRO_YEARLY = settings.RAZORPAY_PLAN_PRO_YEARLY
RAZORPAY_PLAN_PREMIUM_YEARLY = settings.RAZORPAY_PLAN_PREMIUM_YEARLY
RZP_TEST_MODE = settings.RZP_TEST_MODE
