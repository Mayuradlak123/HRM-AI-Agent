import logging
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from typing import Dict, Any, Optional

from core.middleware import get_current_user

logger = logging.getLogger("hrm_agent")

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

web_router = APIRouter(tags=["Web Pages"])

@web_router.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    """Home page"""
    try:
        logger.debug("Serving home page")
        return templates.TemplateResponse("home.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving home page: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@web_router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page"""
    try:
        logger.debug("Serving login page")
        return templates.TemplateResponse("login.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving login page: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@web_router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """Registration page"""
    try:
        logger.debug("Serving register page")
        return templates.TemplateResponse("register.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving register page: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@web_router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Simple chat page - requires authentication"""
    try:
        logger.debug("Serving simple chat page")
        return templates.TemplateResponse("chat.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving chat page: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@web_router.get("/chat-history", response_class=HTMLResponse)
async def chat_history_page(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Advanced chat page with history - requires authentication"""
    try:
        logger.debug(f"Serving chat history page for user {current_user.get('user_id')}")
        return templates.TemplateResponse("chat_with_history.html", {
            "request": request,
            "user_info": {
                "first_name": current_user.get("first_name", "User"),
                "last_name": current_user.get("last_name", ""),
                "role": current_user.get("role", "employee"),
                "user_id": current_user.get("user_id")
            }
        })
    except Exception as e:
        logger.error(f"Error serving chat history page: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@web_router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Dashboard page - requires authentication"""
    try:
        logger.debug("Serving dashboard page")
        return templates.TemplateResponse("dashboard.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving dashboard page: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@web_router.get("/knowledge", response_class=HTMLResponse)
async def knowledge_page(request: Request):
    """Knowledge base page - requires HR/Admin role"""
    try:
        logger.debug("Serving knowledge page")
        return templates.TemplateResponse("knowledge.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving knowledge page: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@web_router.get("/neural-network", response_class=HTMLResponse)
async def neural_network_page(request: Request):
    """Neural Network Deepfake Detector page"""
    try:
        logger.debug("Serving neural network deepfake detector page")
        return templates.TemplateResponse("deepfake_detector.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving neural network page: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@web_router.get("/deepfake-history", response_class=HTMLResponse)
async def deepfake_history_page(request: Request):
    """Deepfake Analysis History page"""
    try:
        logger.debug("Serving deepfake analysis history page")
        return templates.TemplateResponse("deepfake_history.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving deepfake history page: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
