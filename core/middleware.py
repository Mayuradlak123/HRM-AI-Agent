# core/middleware.py
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from core.auth import AuthManager
from config.database import get_database
from models.index import UserRole
from config.logger import logger

# IMPORTANT: make bearer optional so we can fall back to cookies
security = HTTPBearer(auto_error=False)

class AuthMiddleware:
    """Authentication middleware for protecting routes"""

    @staticmethod
    async def get_current_user(
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    ) -> Dict[str, Any]:
        """Accept Bearer token OR 'access_token' cookie."""
        try:
            token: Optional[str] = None

            # Priority 1: Authorization header
            if credentials and credentials.scheme.lower() == "bearer":
                token = credentials.credentials

            # Priority 2: Cookie fallback (for full page refresh / SSR)
            if not token:
                token = request.cookies.get("access_token")

            if not token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Not authenticated"
                )

            payload = AuthManager.get_current_user_payload(token)
            logger.info("Payload ",payload)
            # Verify user exists in DB and is active
            db = get_database()
            user = db["users"].find_one({"user_id": payload["user_id"], "is_active": True})
            logger.info("User ",user)
            
            if not user:
                logger.warning(f"User not found or inactive: {payload['user_id']}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found or inactive"
                )

            # Optional: update last_login timestamp
            db["users"].update_one(
                {"user_id": payload["user_id"]},
                {"$set": {"last_login": datetime.utcnow()}}
            )

            logger.debug(f"Current user authenticated: {payload['user_id']}")
            return payload

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting current user: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )

    @staticmethod
    async def require_role(required_roles: list, current_user: Dict[str, Any]) -> Dict[str, Any]:
        """Require specific user roles"""
        try:
            user_role = current_user.get("role")
            if user_role not in required_roles:
                logger.warning(f"Insufficient permissions. User role: {user_role}, Required: {required_roles}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )
            logger.debug(f"Role check passed for user: {current_user['user_id']}")
            return current_user

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error checking user role: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Role verification failed"
            )

# Dependency wrappers
async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    return await AuthMiddleware.get_current_user(request, credentials)

async def require_hr_role(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    return await AuthMiddleware.require_role([UserRole.HR, UserRole.ADMIN], current_user)

async def require_admin_role(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    return await AuthMiddleware.require_role([UserRole.ADMIN], current_user)

async def require_employee_or_hr(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    return await AuthMiddleware.require_role([UserRole.EMPLOYEE, UserRole.HR, UserRole.ADMIN], current_user)
