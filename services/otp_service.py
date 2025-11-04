import logging
import random
import string
import smtplib
from datetime import datetime, timedelta

from typing import Optional

from config.database import get_database
from config.logger import logger

class OTPService:
    """Service for handling OTP generation, validation, and email sending"""
    
    @staticmethod
    def generate_otp() -> str:
        """Generate a random OTP"""
        try:
            otp = ''.join(random.choices(string.digits, k=6))
            logger.debug(f"OTP generated with length: 6")
            return otp
        except Exception as e:
            logger.error(f"Error generating OTP: {e}")
            raise
    

    @staticmethod
    def store_otp(email: str, otp: str, purpose: str = "login") -> bool:
        """Store OTP in database with expiry"""
        try:
            logger.debug(f"Storing OTP for email: {email}")
            
            db = get_database()
            expiry_time = datetime.utcnow() + timedelta(minutes=10)
            
            # Remove any existing OTPs for this email and purpose
            db["otps"].delete_many({"email": email, "purpose": purpose})
            
            # Store new OTP
            otp_doc = {
                "email": email,
                "otp": otp,
                "purpose": purpose,
                "created_at": datetime.utcnow(),
                "expires_at": expiry_time,
                "is_used": False
            }
            
            db["otps"].insert_one(otp_doc)
            logger.debug(f"OTP stored successfully for: {email}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing OTP for {email}: {e}")
            return False
    
    @staticmethod
    def verify_otp(email: str, otp: str, purpose: str = "login") -> bool:
        """Verify OTP for given email and purpose"""
        try:
            logger.debug(f"Verifying OTP for email: {email}")
            db = get_database()
            current_time = datetime.utcnow()
            
            # Find valid OTP
            otp_doc = db["otps"].find_one({
                "email": email,
                "otp": otp,
            })
            if not otp_doc:
                logger.warning(f"Invalid or expired OTP for: {email}")
                return False
            
            # Mark OTP as used
            db["otps"].update_one(
                {"_id": otp_doc["_id"]},
                {"$set": {"is_used": True, "used_at": current_time}}
            )
            
            logger.info(f"OTP verified successfully for: {email}")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying OTP for {email}: {e}")
            return False
    
    @staticmethod
    def cleanup_expired_otps() -> int:
        """Clean up expired OTPs from database"""
        try:
            db = get_database()
            current_time = datetime.utcnow()
            
            result = db["otps"].delete_many({
                "expires_at": {"$lt": current_time}
            })
            
            deleted_count = result.deleted_count
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} expired OTPs")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up expired OTPs: {e}")
            return 0
