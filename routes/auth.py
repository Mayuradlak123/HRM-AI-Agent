from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, status, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr

from core.auth import AuthManager
from core.middleware import get_current_user
from config.database import get_database
from models.index import User, TokenResponse, UserRole
from config.logger import logger
from services.otp_service import OTPService
auth_router = APIRouter(prefix="/auth", tags=["Authentication"])
# Shared cookie options
COOKIE_KWARGS = dict(
    httponly=True,
    secure=False,     # set True in production (HTTPS)
    samesite="lax",
    max_age=60 * 60 * 24 * 7,  # 7 days
)

# ---------- Schemas ----------
class SendOTPRequest(BaseModel):
    email: EmailStr
    purpose: str = "login"  # login, register

class VerifyOTPRequest(BaseModel):
    email: EmailStr
    otp: str
    purpose: str = "login"

class RegisterWithOTP(BaseModel):
    email: EmailStr
    otp: str
    first_name: str
    last_name: str
    role: UserRole = UserRole.EMPLOYEE
    department_id: Optional[str] = None

class LoginWithOTP(BaseModel):
    email: EmailStr
    otp: str

# ---------- OTP endpoints ----------
@auth_router.post("/send-otp")
async def send_otp(request: SendOTPRequest):
    try:
        logger.info(f"OTP request for {request.purpose}: {request.email}")
        db = get_database()

        if request.purpose == "register":
            if db["users"].find_one({"email": request.email}):
                raise HTTPException(status_code=400, detail="User with this email already exists")
        elif request.purpose == "login":
            if not db["users"].find_one({"email": request.email, "is_active": True}):
                raise HTTPException(status_code=404, detail="User not found")
        else:
            raise HTTPException(status_code=400, detail="Invalid purpose")

        otp = OTPService.generate_otp()
        if not OTPService.store_otp(request.email, otp, request.purpose):
            raise HTTPException(status_code=500, detail="Failed to generate OTP")

        # FOR TESTING ONLY – include otp in response
        return JSONResponse(status_code=200, content={
            "status": "success",
            "message": "OTP sent successfully",
            "email": request.email,
            "otp": otp,
            "testing_note": "OTP is included in response for testing purposes"
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending OTP: {e}")
        raise HTTPException(status_code=500, detail="Failed to send OTP")

@auth_router.post("/verify-otp")
async def verify_otp(request: VerifyOTPRequest):
    try:
        if not OTPService.verify_otp(request.email, request.otp, request.purpose):
            raise HTTPException(status_code=400, detail="Invalid or expired OTP")
        return {"status": "success", "message": "OTP verified successfully", "email": request.email}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying OTP: {e}")
        raise HTTPException(status_code=500, detail="OTP verification failed")

# ---------- Register / Login ----------
@auth_router.post("/register")
async def register_user(user_data: RegisterWithOTP):
    try:
        # Verify OTP
        if not OTPService.verify_otp(user_data.email, user_data.otp, "register"):
            raise HTTPException(status_code=400, detail="Invalid or expired OTP")

        db = get_database()
        if db["users"].find_one({"email": user_data.email}):
            raise HTTPException(status_code=400, detail="User with this email already exists")

        user = User(
            email=user_data.email,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            role=user_data.role,
            department_id=user_data.department_id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db["users"].insert_one(user.dict())

        token_data = {
            "user_id": user.user_id,
            "email": user.email,
            "role": user.role.value,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "department_id": user.department_id
        }
        access_token = AuthManager.create_access_token(token_data)
        refresh_token = AuthManager.create_refresh_token(token_data)

        resp = JSONResponse(status_code=200, content={
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": 1800,
            "user_info": token_data
        })
        resp.set_cookie("access_token", access_token, **COOKIE_KWARGS)
        resp.set_cookie("refresh_token", refresh_token, **COOKIE_KWARGS)
        return resp

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during user registration: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@auth_router.post("/login")
async def login_user(credentials: LoginWithOTP):
    try:
        if not OTPService.verify_otp(credentials.email, credentials.otp, "login"):
            raise HTTPException(status_code=401, detail="Invalid or expired OTP")

        db = get_database()
        user = db["users"].find_one({"email": credentials.email, "is_active": True})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")

        db["users"].update_one({"user_id": user["user_id"]}, {"$set": {"last_login": datetime.utcnow()}})

        token_data = {
            "user_id": user["user_id"],
            "email": user["email"],
            "role": user["role"],
            "first_name": user["first_name"],
            "last_name": user["last_name"],
            "department_id": user.get("department_id")
        }
        access_token = AuthManager.create_access_token(token_data)
        refresh_token = AuthManager.create_refresh_token(token_data)

        resp = JSONResponse(status_code=200, content={
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": 1800,
            "user_info": token_data
        })
        resp.set_cookie("access_token", access_token, **COOKIE_KWARGS)
        resp.set_cookie("refresh_token", refresh_token, **COOKIE_KWARGS)
        return resp

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during login: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

# ---------- Refresh / Me / Logout ----------
@auth_router.post("/refresh")
async def refresh_token(request: Request, refresh_token: Optional[str] = None):
    try:
        token = refresh_token or request.cookies.get("refresh_token")
        if not token:
            raise HTTPException(status_code=401, detail="No refresh token")

        payload = AuthManager.verify_token(token, "refresh")
        new_access = AuthManager.create_access_token({
            "user_id": payload["user_id"],
            "email": payload["email"],
            "role": payload["role"],
            "first_name": payload["first_name"],
            "last_name": payload["last_name"],
            "department_id": payload.get("department_id")
        })

        resp = JSONResponse(status_code=200, content={
            "access_token": new_access,
            "token_type": "bearer",
            "expires_in": 1800
        })
        resp.set_cookie("access_token", new_access, **COOKIE_KWARGS)  # rotate
        return resp

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing token: {e}")
        raise HTTPException(status_code=500, detail="Token refresh failed")

@auth_router.get("/me")
async def get_current_user_info(current_user: Dict[str, Any] = Depends(get_current_user)):
    try:
        db = get_database()
        user = db["users"].find_one({"user_id": current_user["user_id"]}, {"_id": 0})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return {"status": "success", "data": user}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get user information")

@auth_router.post("/logout")
async def logout_user(current_user: Dict[str, Any] = Depends(get_current_user)):
    try:
        resp = JSONResponse(status_code=200, content={"status": "success", "message": "Logged out successfully"})
        resp.delete_cookie("access_token")
        resp.delete_cookie("refresh_token")
        return resp
    except Exception as e:
        logger.error(f"Error during logout: {e}")
        raise HTTPException(status_code=500, detail="Logout failed")
