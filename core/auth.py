# core/auth.py
import os
import time
import uuid
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

import jwt  # PyJWT

# ENV / defaults
JWT_SECRET = os.getenv("JWT_SECRET", "dev-insecure-secret-change-me")
JWT_REFRESH_SECRET = os.getenv("JWT_REFRESH_SECRET", "dev-insecure-refresh-secret-change-me")
JWT_ALG = os.getenv("JWT_ALG", "HS256")
ACCESS_EXPIRE_SECONDS = int(os.getenv("ACCESS_EXPIRE_SECONDS", "86400"))  # 24 hours
REFRESH_EXPIRE_SECONDS = int(os.getenv("REFRESH_EXPIRE_SECONDS", "1209600"))  # 14 days

class AuthManager:
    @staticmethod
    def _now_ts() -> int:
        return int(time.time())

    @staticmethod
    def create_access_token(data: Dict[str, Any]) -> str:
        payload = {
            **data,
            "type": "access",
            "iat": AuthManager._now_ts(),
            "exp": AuthManager._now_ts() + ACCESS_EXPIRE_SECONDS,
            "jti": str(uuid.uuid4()),
        }
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

    @staticmethod
    def create_refresh_token(data: Dict[str, Any]) -> str:
        payload = {
            **data,
            "type": "refresh",
            "iat": AuthManager._now_ts(),
            "exp": AuthManager._now_ts() + REFRESH_EXPIRE_SECONDS,
            "jti": str(uuid.uuid4()),
        }
        return jwt.encode(payload, JWT_REFRESH_SECRET, algorithm=JWT_ALG)

    @staticmethod
    def verify_token(token: str, expected_type: str = "access") -> Dict[str, Any]:
        secret = JWT_SECRET if expected_type == "access" else JWT_REFRESH_SECRET
        payload = jwt.decode(token, secret, algorithms=[JWT_ALG])
        if payload.get("type") != expected_type:
            raise jwt.InvalidTokenError("Token type mismatch")
        return payload

    @staticmethod
    def get_current_user_payload(token: str) -> Dict[str, Any]:
        # legacy helper used by your middleware
        return AuthManager.verify_token(token, "access")
