# enhanced_graph_agent.py
import os
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List, TypedDict, Optional
import json

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from sentence_transformers import SentenceTransformer
from utils.index import get_token_ids,get_tokens

# Option 1: Use transformers with a local model (Recommended)
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Option 2: Alternative - Use Ollama (if installed locally)
# import requests

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

def initialize_llm():
    """Initialize the chosen free LLM"""
    global _llm_client, _tokenizer, _model
    
    if LLM_OPTION == "transformers":
        try:
            # Option 1: Local transformers model (No API key needed)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME)
            _model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_NAME)
            _model.to(device)
            
            # Add padding token if not present
            if _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token
            
            logger.info(f"Initialized local model: {LOCAL_MODEL_NAME} on {device}")
        except Exception as e:
            logger.error(f"Failed to initialize transformers model: {e}")
            _llm_client = None
    
    elif LLM_OPTION == "huggingface":
        try:
            # Option 2: Hugging Face Inference API (free tier, no API key for some models)
            from huggingface_hub import InferenceClient
            _llm_client = InferenceClient()
            logger.info("Initialized Hugging Face Inference Client")
        except Exception as e:
            logger.error(f"Failed to initialize Hugging Face client: {e}")
            _llm_client = None

# Initialize on import
initialize_llm()

connect_to_pinecone(index_name=PINECONE_INDEX)

def embed_query(text: str) -> List[float]:
    return _embed_model.encode(text, normalize_embeddings=True).tolist()

class ChatState(TypedDict, total=False):
    user_id: str
    session_id: str
    question: str
    user_context: Dict[str, Any]  # Employee details, role, department
    ctx: List[Dict[str, Any]]
    database_results: List[Dict[str, Any]]
    rag_answer: str
    final_answer: str
    message_id: str
    query_type: str
    confidence_score: float

# Database Query Functions (unchanged)
def get_employee_context(user_id: str) -> Dict[str, Any]:
    """Get employee details for personalized responses"""
    try:
        db = get_database()
        employee = db["employees"].find_one({"user_id": user_id}, {"_id": 0})
        if not employee:
            return {}
        
        # Get additional context based on employee role/department
        context = {
            "employee": employee,
            "department_policies": [],
            "role_specific_info": []
        }
        
        # Get department-specific policies
        if employee.get("department"):
            dept_policies = list(db["policies"].find({
                "$or": [
                    {"applicable_departments": employee["department"]},
                    {"applicable_departments": "all"}
                ]
            }, {"_id": 0}))
            context["department_policies"] = dept_policies
        
        # Get role-specific information
        if employee.get("role"):
            role_info = list(db["role_information"].find({
                "$or": [
                    {"applicable_roles": employee["role"]},
                    {"applicable_roles": "all"}
                ]
            }, {"_id": 0}))
            context["role_specific_info"] = role_info
            
        return context
    except Exception as e:
        logger.error(f"Error getting employee context: {e}")
        return {}

def query_database_dynamically(question: str, user_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Dynamically query database based on question intent and user context"""
    try:
        db = get_database()
        results = []
        
        # Determine query type based on question content
        question_lower = question.lower()
        employee = user_context.get("employee", {})
        
        # Salary and compensation queries
        if any(term in question_lower for term in ["salary", "pay", "compensation", "wage", "bonus"]):
            if employee.get("employee_id"):
                salary_info = db["salary_information"].find_one({
                    "employee_id": employee["employee_id"]
                }, {"_id": 0})
                if salary_info:
                    results.append({"type": "salary", "data": salary_info})
        
        # Leave and PTO queries
        if any(term in question_lower for term in ["leave", "pto", "vacation", "sick", "holiday", "time off"]):
            leave_policies = list(db["leave_policies"].find({
                "$or": [
                    {"applicable_departments": employee.get("department")},
                    {"applicable_departments": "all"}
                ]
            }, {"_id": 0}))
            results.extend([{"type": "leave_policy", "data": policy} for policy in leave_policies])
            
            # Get user's leave balance
            if employee.get("employee_id"):
                leave_balance = db["leave_balances"].find_one({
                    "employee_id": employee["employee_id"]
                }, {"_id": 0})
                if leave_balance:
                    results.append({"type": "leave_balance", "data": leave_balance})
        
        # Benefits queries
        if any(term in question_lower for term in ["benefit", "insurance", "health", "dental", "vision", "401k", "retirement"]):
            benefits = list(db["benefits"].find({
                "$or": [
                    {"applicable_roles": employee.get("role")},
                    {"applicable_roles": "all"}
                ]
            }, {"_id": 0}))
            results.extend([{"type": "benefit", "data": benefit} for benefit in benefits])
        
        # Company policies
        if any(term in question_lower for term in ["policy", "rule", "guideline", "procedure", "protocol"]):
            policies = list(db["company_policies"].find({
                "$or": [
                    {"applicable_departments": employee.get("department")},
                    {"applicable_departments": "all"}
                ]
            }, {"_id": 0}))
            results.extend([{"type": "policy", "data": policy} for policy in policies])
        
        # Performance and review queries
        if any(term in question_lower for term in ["performance", "review", "evaluation", "feedback", "rating"]):
            if employee.get("employee_id"):
                performance = list(db["performance_reviews"].find({
                    "employee_id": employee["employee_id"]
                }, {"_id": 0}).sort("review_date", -1).limit(3))
                results.extend([{"type": "performance", "data": review} for review in performance])
        
        # Training and development
        if any(term in question_lower for term in ["training", "course", "development", "skill", "certification"]):
            training = list(db["training_programs"].find({
                "$or": [
                    {"target_roles": employee.get("role")},
                    {"target_departments": employee.get("department")},
                    {"target_roles": "all"}
                ]
            }, {"_id": 0}))
            results.extend([{"type": "training", "data": program} for program in training])
        
        # Team and organizational queries
        if any(term in question_lower for term in ["team", "colleague", "manager", "supervisor", "org", "organization"]):
            if employee.get("department"):
                team_info = list(db["employees"].find({
                    "department": employee["department"],
                    "user_id": {"$ne": user_context["employee"]["user_id"]}
                }, {"_id": 0, "user_id": 1, "name": 1, "role": 1, "email": 1}))
                results.extend([{"type": "team", "data": member} for member in team_info])
        
        return results
    except Exception as e:
        logger.error(f"Database query error: {e}")
        return []

def enhanced_rag_retrieval(question: str, user_context: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
    """Enhanced RAG with user context and semantic search"""
    try:
        # Create enhanced query with user context
        employee = user_context.get("employee", {})
        enhanced_query = f"{question}"
        if employee.get("department"):
            enhanced_query += f" department:{employee['department']}"
        if employee.get("role"):
            enhanced_query += f" role:{employee['role']}"
        
        vec = embed_query(enhanced_query)
        connect_to_pinecone(index_name=PINECONE_INDEX)
        idx = get_pinecone_index()
        
        # Query with metadata filters for relevance
        filters = {}
        if employee.get("department"):
            filters["department"] = employee["department"]
        
        res = idx.query(
            vector=vec, 
            top_k=top_k, 
            include_metadata=True,
            filter=filters if filters else None
        )
        
        ctx = []
        for match in getattr(res, "matches", []) or []:
            metadata = getattr(match, "metadata", {}) or {}
            text = metadata.get("text") or metadata.get("content") or ""
            if text:
                ctx.append({
                    "text": text,
                    "score": getattr(match, "score", 0),
                    "metadata": metadata,
                    "relevance": "high" if getattr(match, "score", 0) > 0.8 else "medium"
                })
        
        return sorted(ctx, key=lambda x: x["score"], reverse=True)
    except Exception as e:
        logger.error(f"Enhanced RAG retrieval failed: {e}")
        return []

def generate_with_transformers(prompt: str, max_length: int = 512) -> str:
    """Generate response using local transformers model"""
    try:
        # Encode the prompt
        inputs = _tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=400)
        
        # Generate response
        with torch.no_grad():
            outputs = _model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=_tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.9
            )
        
        # Decode and clean response
        response = _tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt from response
        response = response[len(_tokenizer.decode(inputs[0], skip_special_tokens=True)):].strip()
        
        return response if response else "I understand your question but need more specific information to provide a helpful answer."
    except Exception as e:
        logger.error(f"Transformers generation failed: {e}")
        return "Unable to generate response with local model."

def generate_with_huggingface(prompt: str) -> str:
    """Generate response using Hugging Face Inference API"""
    try:
        # Use a free model from Hugging Face
        response = _llm_client.text_generation(
            prompt,
            model="microsoft/DialoGPT-large",
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True
        )
        return response.strip()
    except Exception as e:
        logger.error(f"Hugging Face generation failed: {e}")
        return "Unable to generate response with Hugging Face model."

def generate_with_ollama(prompt: str) -> str:
    """Generate response using local Ollama (if installed)"""
    try:
        import requests
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama2",  # or "mistral", "codellama", etc.
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "max_tokens": 300
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json().get("response", "No response generated.")
        else:
            return "Ollama service unavailable."
    except Exception as e:
        logger.error(f"Ollama generation failed: {e}")
        return "Unable to connect to Ollama service."

def generate_dynamic_response(question: str, rag_context: List[Dict[str, Any]], 
                            db_results: List[Dict[str, Any]], user_context: Dict[str, Any]) -> str:
    """Generate contextual response using free LLM options"""
    
    employee = user_context.get("employee", {})
    
    # Prepare context for LLM
    context_parts = []
    
    # Add database results
    for result in db_results[:5]:  # Limit to top 5 results
        result_type = result.get("type", "unknown")
        data = result.get("data", {})
        context_parts.append(f"[{result_type.upper()}] {json.dumps(data, default=str)}")
    
    # Add RAG context
    for ctx in rag_context[:3]:  # Limit to top 3 RAG results
        context_parts.append(f"[KNOWLEDGE] {ctx['text']}")
    
    # Construct prompt
    system_context = f"""You are an HRM AI Assistant helping employee {employee.get('name', 'User')} 
from {employee.get('department', 'Unknown')} department with role {employee.get('role', 'Unknown')}.

Context Information:
{chr(10).join(context_parts)}

Question: {question}

Please provide a helpful, professional, and accurate answer based on the context provided."""
    
    # Try different LLM options
    try:
        if LLM_OPTION == "transformers" and _model and _tokenizer:
            return generate_with_transformers(system_context)
        elif LLM_OPTION == "huggingface" and _llm_client:
            return generate_with_huggingface(system_context)
        elif LLM_OPTION == "ollama":
            return generate_with_ollama(system_context)
        else:
            # Fallback to template-based response
            return fallback_response_generation(question, rag_context, db_results, user_context)
    except Exception as e:
        logger.error(f"LLM response generation failed: {e}")
        return fallback_response_generation(question, rag_context, db_results, user_context)

def fallback_response_generation(question: str, rag_context: List[Dict[str, Any]], 
                               db_results: List[Dict[str, Any]], user_context: Dict[str, Any]) -> str:
    """Enhanced fallback response when LLM is not available"""
    if not db_results and not rag_context:
        return "I don't have enough information to answer your question. Please contact HR for assistance."
    
    response_parts = []
    employee = user_context.get("employee", {})
    
    if employee.get("name"):
        response_parts.append(f"Hi {employee['name']},")
    
    # Process database results with better formatting
    for result in db_results[:3]:
        result_type = result.get("type")
        data = result.get("data", {})
        
        if result_type == "salary":
            response_parts.append(f"Your current salary information: Base: ${data.get('base_salary', 'N/A')}, "
                                f"Total: ${data.get('total_compensation', 'N/A')}")
        elif result_type == "leave_balance":
            response_parts.append(f"Your leave balance: Vacation: {data.get('vacation_days', 'N/A')} days, "
                                f"Sick: {data.get('sick_days', 'N/A')} days")
        elif result_type == "leave_policy":
            policy_name = data.get('name', 'Leave Policy')
            response_parts.append(f"{policy_name}: {data.get('description', 'N/A')}")
        elif result_type == "benefit":
            response_parts.append(f"Benefit: {data.get('name', 'N/A')} - {data.get('description', 'N/A')}")
        elif result_type == "policy":
            policy_name = data.get('name', 'Company Policy')
            response_parts.append(f"{policy_name}: {data.get('description', 'N/A')}")
        elif result_type == "training":
            response_parts.append(f"Training: {data.get('name', 'N/A')} - {data.get('description', 'N/A')}")
    
    # Add RAG context with better relevance filtering
    for ctx in rag_context[:2]:
        if ctx.get("score", 0) > 0.7:  # Only high-confidence matches
            response_parts.append(f"Additional information: {ctx['text'][:300]}...")
    
    if not response_parts:
        return "I found some information but couldn't format a specific answer. Please contact HR for detailed assistance."
    
    return "\n\n".join(response_parts)

# Graph Nodes (mostly unchanged, just updated the response generation)
def node_load_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """Load employee context and classify query type"""
    user_context = get_employee_context(state["user_id"])
    
    # Classify query type for better processing
    question_lower = state["question"].lower()
    query_type = "general"
    
    if any(term in question_lower for term in ["salary", "pay", "compensation"]):
        query_type = "compensation"
    elif any(term in question_lower for term in ["leave", "pto", "vacation"]):
        query_type = "leave"
    elif any(term in question_lower for term in ["benefit", "insurance", "health"]):
        query_type = "benefits"
    elif any(term in question_lower for term in ["policy", "rule", "guideline"]):
        query_type = "policy"
    elif any(term in question_lower for term in ["training", "development", "course"]):
        query_type = "training"
    elif any(term in question_lower for term in ["team", "colleague", "manager"]):
        query_type = "organizational"
    
    new_state = dict(state)
    new_state.update({
        "user_context": user_context,
        "query_type": query_type
    })
    return new_state

def node_database_query(state: Dict[str, Any]) -> Dict[str, Any]:
    """Query database for real-time information"""
    db_results = query_database_dynamically(state["question"], state["user_context"])
    
    new_state = dict(state)
    new_state.update({"database_results": db_results})
    return new_state

def node_enhanced_rag(state: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced RAG retrieval with context"""
    ctx = enhanced_rag_retrieval(state["question"], state["user_context"], top_k=10)
    
    # Calculate confidence score based on retrieval quality
    confidence_score = 0.0
    if ctx:
        confidence_score = sum(c.get("score", 0) for c in ctx[:3]) / min(3, len(ctx))
    
    new_state = dict(state)
    new_state.update({
        "ctx": ctx,
        "confidence_score": confidence_score
    })
    return new_state

def node_generate_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate final response using free LLM or fallback"""
    final_answer = generate_dynamic_response(
        state["question"],
        state.get("ctx", []),
        state.get("database_results", []),
        state["user_context"]
    )
    
    new_state = dict(state)
    new_state.update({"final_answer": final_answer})
    return new_state

def node_persist_enhanced(state: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced persistence with metadata"""
    try:
        # Upsert conversation with enhanced metadata
        upsert_enhanced_turn(
            state["session_id"], 
            state["user_id"], 
            "user", 
            state["question"],
            {"query_type": state.get("query_type"), "confidence": state.get("confidence_score")}
        )
        
        upsert_enhanced_turn(
            state["session_id"], 
            state["user_id"], 
            "assistant", 
            state["final_answer"],
            {
                "query_type": state.get("query_type"),
                "confidence": state.get("confidence_score"),
                "sources_used": len(state.get("ctx", [])) + len(state.get("database_results", []))
            }
        )
        
        mid = persist_enhanced_message(
            state["session_id"], 
            state["user_id"], 
            state["question"], 
            state["final_answer"],
            {
                "query_type": state.get("query_type"),
                "confidence_score": state.get("confidence_score"),
                "context_sources": len(state.get("ctx", [])),
                "database_sources": len(state.get("database_results", []))
            }
        )
        
        new_state = dict(state)
        new_state.update({"message_id": mid})
        return new_state
    except Exception as e:
        logger.error(f"Enhanced persist failed: {e}")
        new_state = dict(state)
        new_state.update({"message_id": str(uuid.uuid4())})
        return new_state

def upsert_enhanced_turn(session_id: str, user_id: str, role: str, text: str, metadata: Dict[str, Any] = None):
    """Enhanced turn persistence with metadata"""
    try:
        vec = embed_query(text)
        enhanced_metadata = {
            "type": role,
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "backend": "enhanced-local-free-llm-graph",
            "text": text,
            **(metadata or {})
        }
        
        insert_vector(
            id=str(uuid.uuid4()),
            vector=vec,
            metadata=enhanced_metadata,
        )
    except Exception as e:
        logger.error(f"Enhanced Pinecone upsert error: {e}")

def persist_enhanced_message(session_id: str, user_id: str, user_msg: str, agent_msg: str, metadata: Dict[str, Any] = None) -> str:
    """Enhanced message persistence"""
    db = get_database()
    mid = str(uuid.uuid4())
    token_ids=get_token_ids(user_msg)
    tokens=get_tokens(user_msg)
    message_doc = {
        "message_id": mid,
        "token_ids":token_ids,
        "tokens":tokens,
        "session_id": session_id,
        "user_id": user_id,
        "user_message": user_msg,
        "agent_response": agent_msg,
        "response_metadata": metadata or {},
        "timestamp": datetime.utcnow(),
        "processed_at": datetime.utcnow(),
        "version": "enhanced-free-llm"
    }
    
    db["chat_messages"].insert_one(message_doc)
    db["chat_sessions"].update_one(
        {"session_id": session_id}, 
        {"$set": {"updated_at": datetime.utcnow()}}
    )
    return mid

def ensure_session(user_id: str, session_id: Optional[str]) -> str:
    """Enhanced session management"""
    db = get_database()
    if session_id:
        db["chat_sessions"].update_one(
            {"session_id": session_id, "user_id": user_id},
            {
                "$set": {"updated_at": datetime.utcnow()}, 
                "$setOnInsert": {
                    "is_active": True, 
                    "title": "New Chat",
                    "created_at": datetime.utcnow(),
                    "version": "enhanced-free-llm"
                }
            },
            upsert=True
        )
        return session_id
    
    sid = str(uuid.uuid4())
    db["chat_sessions"].insert_one({
        "session_id": sid, 
        "user_id": user_id, 
        "title": "New Chat",
        "is_active": True, 
        "created_at": datetime.utcnow(), 
        "updated_at": datetime.utcnow(),
        "version": "enhanced-free-llm"
    })
    return sid

# Build Enhanced Graph
def build_enhanced_graph():
    """Build enhanced processing graph with real-time capabilities"""
    g = StateGraph(dict)
    
    # Add nodes
    g.add_node("load_context", RunnableLambda(node_load_context))
    g.add_node("database_query", RunnableLambda(node_database_query))
    g.add_node("enhanced_rag", RunnableLambda(node_enhanced_rag))
    g.add_node("generate_response", RunnableLambda(node_generate_response))
    g.add_node("persist", RunnableLambda(node_persist_enhanced))
    
    # Define flow
    g.set_entry_point("load_context")
    g.add_edge("load_context", "database_query")
    g.add_edge("database_query", "enhanced_rag")
    g.add_edge("enhanced_rag", "generate_response")
    g.add_edge("generate_response", "persist")
    g.add_edge("persist", END)
    
    return g.compile()

# Initialize enhanced graph
ENHANCED_GRAPH = build_enhanced_graph()

# Real-time processing function
def process_realtime_query(user_id: str, session_id: Optional[str], question: str) -> Dict[str, Any]:
    """Process query with real-time database lookups and enhanced RAG"""
    logger.info(f"[realtime] Processing query for user_id={user_id}")
    
    try:
        sid = ensure_session(user_id, session_id)
        
        init_state: ChatState = {
            "user_id": user_id,
            "session_id": sid,
            "question": question
        }
        
        # Run enhanced graph
        result: ChatState = ENHANCED_GRAPH.invoke(init_state)
        
        return {
            "response": result.get("final_answer", "Unable to generate response."),
            "session_id": result.get("session_id", sid),
            "message_id": result.get("message_id", str(uuid.uuid4())),
            "metadata": {
                "query_type": result.get("query_type", "general"),
                "confidence_score": result.get("confidence_score", 0.0),
                "context_snippets": result.get("ctx", [])[:3],
                "database_results_count": len(result.get("database_results", [])),
                "processing_time": datetime.utcnow().isoformat(),
                "llm_backend": LLM_OPTION
            }
        }
    except Exception as e:
        logger.error(f"Real-time processing failed: {e}")
        raise