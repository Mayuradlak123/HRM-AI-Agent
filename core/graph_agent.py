# enhanced_graph_agent.py
import os
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List, TypedDict, Optional
import json


from sentence_transformers import SentenceTransformer
from utils.index import get_token_ids,get_tokens

# Option 3: Alternative - Use Hugging Face Inference API (free tier)
# from huggingface_hub import InferenceClient

from config.database import get_database
from config.pinecone import connect_to_pinecone, get_pinecone_index, insert_vector

from config.logger import logger
# Configuration
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "hrm-knowledge")

# Free LLM Configuration Options
LLM_OPTION = os.getenv("LLM_OPTION", "transformers")  # "transformers", "ollama", or "huggingface"
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "microsoft/DialoGPT-medium")  # Lightweight option
# For better performance, you could use: "microsoft/DialoGPT-large" or "facebook/blenderbot-400M-distill"

# Initialize models
_embed_model = SentenceTransformer(LOCAL_EMBED_MODEL)

# Initialize LLM based on chosen option
_llm_client = None
_tokenizer = None
_model = None


def embed_query(text: str) -> List[float]:
    return _embed_model.encode(text, normalize_embeddings=True).tolist()
