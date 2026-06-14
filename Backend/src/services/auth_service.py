import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import HTTPException, status
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# Import settings from config
from src.core.config import MONGODB_URL, MONGODB_DB_NAME, MONGODB_CONNECT_TIMEOUT, SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from src.core.models import User

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# MongoDB connection with error handling
try:
    client = AsyncIOMotorClient(
        MONGODB_URL,
        serverSelectionTimeoutMS=MONGODB_CONNECT_TIMEOUT
    )
    # Defer connection verification until the first actual DB operation to avoid
    # requiring a running event loop at import time.
    print(f"Configured MongoDB client for {MONGODB_URL}")
    
    db = client[MONGODB_DB_NAME]
    users_collection = db.users
except Exception as e:
    print(f"MongoDB connection error: {e}")
    print("WARNING: Authentication service will not work until MongoDB is available")
    # We'll initialize these as None and check before each operation
    client = None
    db = None
    users_collection = None

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


async def get_user_by_email(email: str):
    if users_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )
    
    try:
        user = await users_collection.find_one({"email": email})
        return user
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )


async def create_user(
    email: str,
    password: str,
    name: Optional[str] = None,
    role: str = "student",
    institute: Optional[str] = None,
    preferred_language: Optional[str] = None
):
    if users_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )

    try:
        # Check if user already exists
        existing_user = await get_user_by_email(email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        # Validate role. Public signup is always a student; privileged roles
        # require admin/sub-admin enrollment. Sanitize any caller-supplied role.
        allowed_roles = {"student", "teacher", "subadmin", "admin"}
        if role not in allowed_roles:
            role = "student"
        if role in {"admin", "subadmin", "teacher"}:
            role = "student"

        # Create new user with hashed password
        hashed_password = get_password_hash(password)
        user = User(
            email=email,
            password_hash=hashed_password,
            name=name,
            role=role,  # type: ignore[arg-type]
            institute=institute,
            preferred_language=preferred_language or "en",
        )

        result = await users_collection.insert_one(user.model_dump(by_alias=True))
        created_user = await users_collection.find_one({"_id": result.inserted_id})
        return created_user
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )


async def authenticate_user(email: str, password: str):
    if users_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )
    
    try:
        user = await get_user_by_email(email)
        if not user:
            return False
        if not verify_password(password, user["password_hash"]):
            return False
        return user
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt