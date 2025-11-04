from config.logger import logger
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from services.mistralai import get_mistralai_service,correct_grammar_service
from pydantic import BaseModel
mistral_router = APIRouter(prefix="/mistralai", tags=["MistralAI"])
import datetime
from transformers import pipeline
  
class MistralAIGenerateRequest(BaseModel):
    prompt: str

@mistral_router.post("/generate")
async def generate_text(request: MistralAIGenerateRequest):
    prompt = request.prompt
    response = get_mistralai_service(prompt)
    return JSONResponse(content={"message": response, "status": "success", "status_code": 200, "timestamp": datetime.datetime.now().isoformat()})


@mistral_router.post("/correct-grammar")
async def correct_grammar(request: MistralAIGenerateRequest):
    prompt = request.prompt
    response = correct_grammar_service(prompt)
    return JSONResponse(content={"message": response, "status": "success", "status_code": 200, "timestamp": datetime.now().isoformat()})