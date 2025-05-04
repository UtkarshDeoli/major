import os
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # MongoDB settings
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "phadai"
    MONGODB_CONNECT_TIMEOUT: int = 30000  # 30 seconds timeout
    
    # JWT settings
    SECRET_KEY: str = "thisisasupersecretkeymadebyutkarsh"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
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