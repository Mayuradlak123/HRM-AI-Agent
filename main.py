
from pathlib import Path

from fastapi import FastAPI,UploadFile,File, Request,HTTPException as FastAPIHTTPException
from fastapi.responses import JSONResponse
# from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from config.database import connect_to_mongo, close_mongo_connection
from config.pinecone import connect_to_pinecone, close_pinecone_connection
from routes.web import web_router
from routes.auth import auth_router
from routes.chat import chat_router
# from routes.image import ocr_router
# from routes.neural_networks import neural_router
# from db.seed_data import seed_database
from pydantic import BaseModel
import os
import traceback
import torch.nn.functional as F
import torch
# from db.logger import logger
# from utils.index import process_etl_file,get_etl_health_check
# Optional tokenizer - only load if needed
tokenizer = None

from fastapi.staticfiles import StaticFiles

# DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "data.json"

@asynccontextmanager
async def lifespan(app: FastAPI):
    connect_to_mongo()
    connect_to_pinecone()  # Use index name from environment variable
    # seed_database()
    yield
    close_pinecone_connection()
    close_mongo_connection()

app = FastAPI(
    title="HRM Agent API",
    lifespan=lifespan
)

# Include web routes (no prefix)
app.include_router(web_router)

# Include API routes with /api prefix
app.include_router(auth_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
# app.include_router(torch_router, prefix="/api")
# app.include_router(neural_router, prefix="/api")
# app.include_router(mistral_router, prefix="/api")
# app.include_router(ocr_router, prefix="/api")
# app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
@asynccontextmanager
async def lifespan(app: FastAPI):
    connect_to_mongo()
    connect_to_pinecone()  # Use index name from environment variable
    # seed_database()
    yield
    close_pinecone_connection()
    close_mongo_connection()