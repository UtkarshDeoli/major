from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import HTTPException, status
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from pymongo.errors import ConnectionFailure, DuplicateKeyError, ServerSelectionTimeoutError

# Import settings from config
from src.core.config import MONGODB_URL, MONGODB_DB_NAME, MONGODB_CONNECT_TIMEOUT, SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from src.core.models import User

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# MongoDB connection (initialized lazily on first use so the client is bound to
# the event loop that actually performs I/O, e.g. the loop created by TestClient).
client = None
db = None
users_collection = None


def _ensure_auth_db():
    """Create the MongoDB client and users collection if not already present."""
    global client, db, users_collection
    if users_collection is not None:
        return

    try:
        client = AsyncIOMotorClient(
            MONGODB_URL,
            serverSelectionTimeoutMS=MONGODB_CONNECT_TIMEOUT
        )
        db = client[MONGODB_DB_NAME]
        users_collection = db.users
        # Ensure a unique index on email to prevent duplicate users (TOCTOU race)
        import asyncio
        asyncio.ensure_future(users_collection.create_index("email", unique=True, background=True))
    except (ConnectionFailure, ServerSelectionTimeoutError, ValueError) as e:
        print(f"MongoDB connection error: {e}")
        print("WARNING: Authentication service will not work until MongoDB is available")
        client = None
        db = None
        users_collection = None


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


async def get_user_by_email(email: str):
    _ensure_auth_db()
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
    preferred_language: Optional[str] = None,
    curriculum: Optional[str] = None,
):
    _ensure_auth_db()
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
            auth_provider="email",
            curriculum=curriculum,
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
    _ensure_auth_db()
    if users_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )

    try:
        user = await get_user_by_email(email)
        if not user:
            return False
        # If user has no password_hash (Google-only account), password auth fails
        if not user.get("password_hash"):
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


async def find_or_create_google_user(email: str, name: Optional[str], google_sub: str):
    """Find a user by email and link Google auth, or create a new Google user.

    - If a user with this email exists: update provider_uid and auth_provider (merge).
    - If no user exists: create a new one with auth_provider="google".

    Returns the user document dict.
    """
    _ensure_auth_db()
    if users_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )

    try:
        existing = await get_user_by_email(email)

        if existing:
            # Merge: link Google auth to existing account
            update_fields = {
                "auth_provider": "google",
                "provider_uid": google_sub,
                "updated_at": datetime.now(timezone.utc),
            }
            if name and not existing.get("name"):
                update_fields["name"] = name
            await users_collection.update_one(
                {"_id": existing["_id"]},
                {"$set": update_fields}
            )
            return await users_collection.find_one({"_id": existing["_id"]})

        # Create new Google user
        user = User(
            email=email,
            name=name,
            auth_provider="google",
            provider_uid=google_sub,
            role="student",
        )
        result = await users_collection.insert_one(user.model_dump(by_alias=True))
        return await users_collection.find_one({"_id": result.inserted_id})

    except HTTPException:
        raise
    except DuplicateKeyError:
        # TOCTOU race: another request created this user between our check and insert
        return await get_user_by_email(email)
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
