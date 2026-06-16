from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import jwt

from src.core.config import SECRET_KEY, ALGORITHM
from src.services.auth_service import get_user_by_email

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


def _decode_token(token: str) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.PyJWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id: str | None = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing subject",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id


async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """Extract and validate the current user email from a JWT bearer token."""
    return _decode_token(token)


async def get_current_user_optional(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[str]:
    """Extract the current user email if a valid token is provided, else None."""
    if token is None:
        return None
    try:
        return _decode_token(token)
    except HTTPException:
        return None


async def get_current_user_with_role(
    token: str = Depends(oauth2_scheme),
) -> dict:
    """Return the current authenticated user document plus their email."""
    email = _decode_token(token)
    user = await get_user_by_email(email)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User no longer exists",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"email": email, "user": user}


def require_role(*allowed_roles: str):
    """Dependency factory that restricts access to the given roles.

    Usage:
        @router.get("/teacher-only")
        async def teacher_only(user=Depends(require_role("teacher"))):
            ...
    """
    async def _role_checker(
        user_info: dict = Depends(get_current_user_with_role),
    ) -> dict:
        user = user_info["user"]
        role = user.get("role", "student")
        if role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to access this resource",
            )
        return user_info

    return _role_checker
