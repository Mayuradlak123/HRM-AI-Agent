# enhanced_chat.py
import uuid
import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
from io import BytesIO
from bson import ObjectId

from sklearn.metrics.pairwise import cosine_similarity
from services.mistralai import get_mistralai_service,get_mistral_with_context_service
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks,status,Body
from fastapi.responses import JSONResponse, StreamingResponse
import pandas as pd
import PyPDF2
from sentence_transformers import SentenceTransformer
from pymongo.errors import PyMongoError

from core.middleware import require_employee_or_hr
from config.database import get_database
from models.index import ChatRequest, ChatResponse, KnowledgeDocument, DeleteChatRequest
from config.pinecone import insert_vector, get_pinecone_index,query_vector
from core.graph_agent import process_realtime_query, embed_query
from core.graph_agent import enhanced_rag_retrieval, get_employee_context
# from services.hrm_agent_service import get_hrm_agent_service
from services.get_user_data import get_user_hr_data
from config.logger import logger
# Pinecone metadata only supports strings, numbers, booleans, or lists of strings.
def _sanitize_pinecone_metadata(md: Dict[str, Any]) -> Dict[str, Any]:
    safe: Dict[str, Any] = {}
    for k, v in (md or {}).items():
        if v is None:
            safe[k] = "unknown"
        elif isinstance(v, (str, bool, int, float)):
            safe[k] = v
        elif isinstance(v, list):
            # Pinecone allows list of strings only; coerce other types to strings and drop None
            safe[k] = [str(x) for x in v if x is not None]
        else:
            # Fallback to string representation
            safe[k] = str(v)
    return safe
chat_router = APIRouter(prefix="/chat", tags=["Enhanced AI Chat"])

# File processing utilities
def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_file = BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return ""

def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from DOCX file"""
    return ""



def extract_text_from_csv(file_content: bytes) -> str:
    """Extract and format text from CSV file"""
    try:
        df = pd.read_csv(BytesIO(file_content))
        # Convert CSV to readable text format
        text = f"CSV Data Summary:\n"
        text += f"Columns: {', '.join(df.columns.tolist())}\n"
        text += f"Rows: {len(df)}\n\n"
        
        # Add sample data (first 5 rows)
        text += "Sample Data:\n"
        text += df.head().to_string(index=False)
        
        return text
    except Exception as e:
        logger.error(f"CSV extraction error: {e}")
        return ""

def process_uploaded_file(file_content: bytes, filename: str, content_type: str) -> str:
    """Process uploaded file and extract text content"""
    text = ""
    
    if content_type == "application/pdf" or filename.lower().endswith('.pdf'):
        text = extract_text_from_pdf(file_content)
    elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or filename.lower().endswith('.docx'):
        text = extract_text_from_docx(file_content)
    elif content_type == "text/csv" or filename.lower().endswith('.csv'):
        text = extract_text_from_csv(file_content)
    elif content_type == "text/plain" or filename.lower().endswith('.txt'):
        text = file_content.decode('utf-8', errors='ignore')
    else:
        raise ValueError(f"Unsupported file type: {content_type}")
    
    return text


@chat_router.post("/message", response_model=ChatResponse)
async def send_enhanced_message(
    chat_request: ChatRequest, 
    current_user: Dict[str, Any] = Depends(require_employee_or_hr)
):
    """Enhanced message processing using Mistral Contextual AI Agent"""
    try:
        # Validate message input
        if not chat_request.message or not chat_request.message.strip():
            logger.warning(f"Empty message from user_id={current_user['user_id']}")
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        db = get_database()
        
        # ✅ Handle session creation/retrieval
        if not chat_request.session_id:
            logger.info("Session title  not exists  ")
            # Create new session with title from first message
            session_id = str(uuid.uuid4())
            session_title = chat_request.message.strip()[:50]  # First 50 chars as title
            
            new_session = {
                "session_id": session_id,
                "user_id": current_user["user_id"],
                "title": session_title,
                "messages": [],
                "is_active": True,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "query_count": 0
            }
            db["chat_sessions"].insert_one(new_session)
            logger.info(f"Created new session {session_id} with title: {session_title}")
        else:
            logger.info("Session title updated  ")

            session_id = chat_request.session_id
            
            # Update session title if it's the first message in existing session
            existing_messages = db["chat_messages"].count_documents({
                "session_id": session_id,
                "user_id": current_user["user_id"]
            })
            
            if existing_messages == 0:
                session_title = chat_request.message.strip()[:50]
                db["chat_sessions"].update_one(
                    {"session_id": session_id, "user_id": current_user["user_id"]},
                    {"$set": {"title": session_title, "updated_at": datetime.utcnow()}}
                )
                logger.info(f"Updated session {session_id} title: {session_title}")

        # ✅ Step 1: Fetch contextual user HR data
        user_data = get_user_hr_data(current_user["user_id"])

        # ✅ Step 2: Send query + HR context to Mistral
        response = get_mistral_with_context_service(
            query=chat_request.message.strip(),
            context_list=user_data
        )

        # ✅ Step 3: Update session with response info
        db["chat_sessions"].update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "updated_at": datetime.utcnow(),
                    "is_active": True,
                    "mistralai_response": response
                },
                "$inc": {"query_count": 1}
            }
        )

        # ✅ Step 4: Store the message exchange
        message_id = str(uuid.uuid4())
        message_data = {
            "message_id": message_id,
            "session_id": session_id,
            "user_id": current_user["user_id"],
            "user_message": chat_request.message.strip(),
            "ai_response": response,
            "timestamp": datetime.utcnow(),
            "response_type": "mistral_contextual"
        }
        db["chat_messages"].insert_one(message_data)
        logger.info(f"Stored message {message_id} in session {session_id}")

        # ✅ Step 5: Generate embeddings and upsert to Pinecone
        try:
            query_text = chat_request.message.strip()
            answer_text = response

            # Create embeddings
            q_vec = embed_query(query_text)
            a_vec = embed_query(answer_text)

            # Prepare metadata
            base_meta = {
                "source": "chat",
                "session_id": session_id,
                "message_id": message_id,
                "user_id": current_user["user_id"],
                "timestamp": datetime.utcnow().isoformat(),
                "response_type": "mistral_contextual"
            }

            insert_vector(
                id=f"chat:{session_id}:{message_id}:q",
                vector=q_vec,
                metadata=_sanitize_pinecone_metadata({
                    **base_meta,
                    "type": "chat_query",
                    "text": query_text or ""
                })
            )
            insert_vector(
                id=f"chat:{session_id}:{message_id}:a",
                vector=a_vec,
                metadata=_sanitize_pinecone_metadata({
                    **base_meta,
                    "type": "chat_answer",
                    "text": answer_text or ""
                })
            )
            logger.info(f"Successfully upserted embeddings to Pinecone for message {message_id}")
        except Exception as e:
            logger.error(f"Pinecone upsert failed for chat message {message_id}: {e}")

        # ✅ Step 6: Return structured response
        return ChatResponse(
            response=response,
            session_id=session_id,
            message_id=message_id,
            metadata={"response_type": "mistral_contextual"},
            mistralai_response=response
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Mistral contextual chat processing failed: {e}")
        raise HTTPException(status_code=500, detail="Chat processing failed")


@chat_router.post("/new-session")
async def create_chat_session(
    session_data: Dict[str, Any] = Body(...),
    current_user: Dict[str, Any] = Depends(require_employee_or_hr)
):
    """Create a new chat session"""
    try:
        db = get_database()
        
        session_id = str(uuid.uuid4())
        title = session_data.get("title", "New Chat")
        
        new_session = {
            "session_id": session_id,
            "user_id": current_user["user_id"],
            "title": title,
            "messages": [],
            "is_active": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "query_count": 0
        }
        
        db["chat_sessions"].insert_one(new_session)
        logger.info(f"Created new chat session {session_id} for user {current_user['user_id']}")
        
        return {
            "status": "success",
            "status_code": status.HTTP_201_CREATED,
            "message": "Session created successfully",
            "session_id": session_id,
            "title": title
        }
        
    except PyMongoError as e:
        logger.error(f"Database error while creating session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error while creating session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )
@chat_router.post("/upload-document")
async def upload_knowledge_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(...),
    category: str = Form(default="general"),
    document_type: str = Form(default="policy"),
    tags: str = Form(default=""),
    current_user: Dict[str, Any] = Depends(require_employee_or_hr)
):
    """Upload and process knowledge document with automatic embedding"""
    if current_user["role"] not in ["hr", "admin"]:
        raise HTTPException(status_code=403, detail="Only HR and Admin users can upload documents")
    
    try:
        # Validate file size (max 10MB)
        if file.size > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size too large (max 10MB)")
        
        # Read file content
        file_content = await file.read()
        
        # Extract text based on file type
        try:
            extracted_text = process_uploaded_file(file_content, file.filename, file.content_type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="No text content could be extracted from the file")
         
        # Create document record
        db = get_database()
        doc_id = str(uuid.uuid4())
        
        document = {
            "document_id": doc_id,
            "title": title,
            "content": extracted_text,
            "document_type": document_type,
            "category": category,
            "tags": [tag.strip() for tag in tags.split(",") if tag.strip()],
            "filename": file.filename,
            "file_size": file.size,
            "content_type": file.content_type,
            "created_by": current_user["user_id"],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "is_active": True,
            "processing_status": "processing"
        }
        
        db["knowledge_documents"].insert_one(document)
        
        # Prepare metadata for embedding
        embedding_metadata = {
            "document_id": doc_id,
            "title": title,
            "category": category,
            "document_type": document_type,
            "tags": document["tags"],
            "created_by": current_user["user_id"],
            "created_at": datetime.utcnow().isoformat(),
            "filename": file.filename
        }
        
        # Schedule background embedding
        await chunk_and_embed_document(extracted_text, embedding_metadata, background_tasks)
        
        # Update processing status
        background_tasks.add_task(
            lambda: db["knowledge_documents"].update_one(
                {"document_id": doc_id},
                {"$set": {"processing_status": "completed"}}
            )
        )
        
        return JSONResponse(
            status_code=201,
            content={
                "status": "success",
                "document_id": doc_id,
                "message": "Document uploaded and is being processed for search",
                "extracted_length": len(extracted_text)
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Document upload failed")


@chat_router.get("/analytics/query-insights")
async def get_query_insights(
    current_user: Dict[str, Any] = Depends(require_employee_or_hr),
    days: int = 30
):
    """Get insights about query patterns and types"""
    try:
        db = get_database()
        
        # Aggregate query types
        pipeline = [
            {
                "$match": {
                    "user_id": current_user["user_id"],
                    "timestamp": {"$gte": datetime.utcnow().replace(day=datetime.utcnow().day - days)}
                }
            },
            {
                "$group": {
                    "_id": "$response_metadata.query_type",
                    "count": {"$sum": 1},
                    "avg_confidence": {"$avg": "$response_metadata.confidence_score"}
                }
            }
        ]
        
        query_stats = list(db["chat_messages"].aggregate(pipeline))
        
        return {
            "status": "success",
            "data": {
                "query_types": query_stats,
                "period_days": days,
                "total_queries": sum(stat["count"] for stat in query_stats)
            }
        }
    except Exception as e:
        logger.error(f"Analytics query failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get query insights")


# Document management endpoints
@chat_router.get("/documents/status")
async def get_document_processing_status(
    current_user: Dict[str, Any] = Depends(require_employee_or_hr)
):
    """Get status of uploaded documents"""
    try:
        db = get_database()
        docs = list(db["knowledge_documents"].find(
            {"created_by": current_user["user_id"]},
            {"_id": 0, "document_id": 1, "title": 1, "processing_status": 1, "created_at": 1}
        ).sort("created_at", -1))
        
        return {"status": "success", "documents": docs}
    except Exception as e:
        logger.error(f"Document status query failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get document status")

@chat_router.delete("/documents/{document_id}")
async def delete_knowledge_document(
    document_id: str,
    current_user: Dict[str, Any] = Depends(require_employee_or_hr)
):
    """Delete a knowledge document and its embeddings"""
    if current_user["role"] not in ["hr", "admin"]:
        raise HTTPException(status_code=403, detail="Only HR and Admin users can delete documents")
    
    try:
        db = get_database()
        
        # Mark document as inactive
        result = db["knowledge_documents"].update_one(
            {"document_id": document_id, "created_by": current_user["user_id"]},
            {"$set": {"is_active": False, "deleted_at": datetime.utcnow()}}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # TODO: Delete from Pinecone (requires namespace or metadata filtering)
        # This would require implementing deletion by metadata filter in your Pinecone setup
        
        return {"status": "success", "message": "Document deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document deletion failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")

# Enhanced session management
@chat_router.get("/sessions/enhanced")
async def get_enhanced_chat_sessions(
    current_user: Dict[str, Any] = Depends(require_employee_or_hr), 
    limit: int = 20,
    include_stats: bool = False
):
    """Get chat sessions with enhanced statistics"""
    try:
        db = get_database()
        
        # Base session query
        sessions = list(db["chat_sessions"].find(
            {"user_id": current_user["user_id"], "is_active": True},
            {"_id": 0}
        ).sort("updated_at", -1).limit(limit))
        
        if include_stats:
            # Add message counts and query type distribution
            for session in sessions:
                session_id = session["session_id"]
                
                # Message count
                message_count = db["chat_messages"].count_documents({"session_id": session_id})
                session["message_count"] = message_count
                
                # Query type distribution
                query_types = list(db["chat_messages"].aggregate([
                    {"$match": {"session_id": session_id}},
                    {"$group": {
                        "_id": "$response_metadata.query_type",
                        "count": {"$sum": 1}
                    }}
                ]))
                session["query_type_distribution"] = query_types
        
        return {
            "status": "success", 
            "data": sessions, 
            "count": len(sessions),
            "enhanced_features": include_stats
        }
    except Exception as e:
        logger.error(f"Error getting enhanced chat sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve enhanced chat sessions")

@chat_router.get("/search/semantic")
async def semantic_search_knowledge(
    query: str,
    current_user: Dict[str, Any] = Depends(require_employee_or_hr),
    limit: int = 10,
    min_score: float = 0.7
):
    """Perform semantic search across knowledge base"""
    try:
        
        user_context = get_employee_context(current_user["user_id"])
        results = enhanced_rag_retrieval(query, user_context, top_k=limit)
        
        # Filter by minimum score
        filtered_results = [r for r in results if r.get("score", 0) >= min_score]
        
        return {
            "status": "success",
            "query": query,
            "results": filtered_results,
            "total_found": len(results),
            "filtered_count": len(filtered_results),
            "min_score_filter": min_score
        }
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        raise HTTPException(status_code=500, detail="Semantic search failed")

# Pinecone embedding retrieval
@chat_router.get("/embedded-conversations")
async def get_embedded_conversations(
    current_user: Dict[str, Any] = Depends(require_employee_or_hr),
    limit: int = 20,
    include_answers: bool = True
):
    """Retrieve embedded chat conversations from Pinecone"""
    try:
        from db.pinecone_database import get_pinecone_index
        
        idx = get_pinecone_index()
        
        # Query for chat conversations for this user
        # We'll use a dummy query vector to get results with metadata filtering
        dummy_vector = embed_query("test query")
        
        # Build filter for user's chat conversations
        filter_dict = {
            "user_id": current_user["user_id"],
            "source": {"$in": ["chat", "chat_stream", "bulk_test"]}
        }
        
        if not include_answers:
            filter_dict["type"] = {"$in": ["chat_query", "bulk_test_query"]}
        
        # Query Pinecone
        results = idx.query(
            vector=dummy_vector,
            top_k=limit * 2 if include_answers else limit,  # Get more if we want both queries and answers
            include_metadata=True,
            filter=filter_dict
        )
        
        # Organize results by session and message
        conversations = {}
        for match in results.matches:
            metadata = match.metadata
            session_id = metadata.get("session_id", "unknown")
            message_id = metadata.get("message_id", "unknown")
            conv_type = metadata.get("type", "unknown")
            
            key = f"{session_id}:{message_id}"
            if key not in conversations:
                conversations[key] = {
                    "session_id": session_id,
                    "message_id": message_id,
                    "timestamp": metadata.get("timestamp", ""),
                    "query_category": metadata.get("query_category", "unknown"),
                    "query_action": metadata.get("query_action", "unknown"),
                    "response_type": metadata.get("response_type", "unknown"),
                    "source": metadata.get("source", "unknown"),
                    "query": None,
                    "answer": None
                }
            
            # Store query or answer text
            if "query" in conv_type:
                conversations[key]["query"] = metadata.get("text", "")
            elif "answer" in conv_type:
                conversations[key]["answer"] = metadata.get("text", "")
        
        # Convert to list and sort by timestamp
        conversation_list = list(conversations.values())
        conversation_list.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return {
            "status": "success",
            "conversations": conversation_list[:limit],
            "total_found": len(conversation_list),
            "user_id": current_user["user_id"],
            "include_answers": include_answers
        }
    
    except Exception as e:
        logger.error(f"Failed to retrieve embedded conversations: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve embedded conversations")

@chat_router.get("/pinecone-stats")
async def get_pinecone_chat_stats(
    current_user: Dict[str, Any] = Depends(require_employee_or_hr)
):
    """Get statistics about embedded chat data in Pinecone"""
    try:
        from db.pinecone_database import get_pinecone_index
        
        idx = get_pinecone_index()
        stats = idx.describe_index_stats()
        
        # Try to get user-specific stats by querying with filters
        dummy_vector = embed_query("test query")
        
        # Count queries for this user
        query_results = idx.query(
            vector=dummy_vector,
            top_k=1,
            include_metadata=True,
            filter={
                "user_id": current_user["user_id"],
                "type": {"$in": ["chat_query", "bulk_test_query"]}
            }
        )
        
        # Count answers for this user
        answer_results = idx.query(
            vector=dummy_vector,
            top_k=1,
            include_metadata=True,
            filter={
                "user_id": current_user["user_id"],
                "type": {"$in": ["chat_answer", "bulk_test_answer"]}
            }
        )
        
        return {
            "status": "success",
            "pinecone_index_stats": {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness
            },
            "user_stats": {
                "user_id": current_user["user_id"],
                "estimated_queries": len(query_results.matches) if query_results else 0,
                "estimated_answers": len(answer_results.matches) if answer_results else 0
            }
        }
    
    except Exception as e:
        logger.error(f"Failed to get Pinecone stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get Pinecone statistics")

# Pinecone embedding retrieval
@chat_router.get("/embedded-conversations")
async def get_embedded_conversations(
    current_user: Dict[str, Any] = Depends(require_employee_or_hr),
    limit: int = 20,
    include_answers: bool = True
):
    """Retrieve embedded chat conversations from Pinecone"""
    try:
        from db.pinecone_database import get_pinecone_index
        
        idx = get_pinecone_index()
        
        # Query for chat conversations for this user
        # We'll use a dummy query vector to get results with metadata filtering
        dummy_vector = embed_query("test query")
        
        # Build filter for user's chat conversations
        filter_dict = {
            "user_id": current_user["user_id"],
            "source": {"$in": ["chat", "chat_stream", "bulk_test"]}
        }
        
        if not include_answers:
            filter_dict["type"] = {"$in": ["chat_query", "bulk_test_query"]}
        
        # Query Pinecone
        results = idx.query(
            vector=dummy_vector,
            top_k=limit * 2 if include_answers else limit,  # Get more if we want both queries and answers
            include_metadata=True,
            filter=filter_dict
        )
        
        # Organize results by session and message
        conversations = {}
        for match in results.matches:
            metadata = match.metadata
            session_id = metadata.get("session_id", "unknown")
            message_id = metadata.get("message_id", "unknown")
            conv_type = metadata.get("type", "unknown")
            
            key = f"{session_id}:{message_id}"
            if key not in conversations:
                conversations[key] = {
                    "session_id": session_id,
                    "message_id": message_id,
                    "timestamp": metadata.get("timestamp", ""),
                    "query_category": metadata.get("query_category", "unknown"),
                    "query_action": metadata.get("query_action", "unknown"),
                    "response_type": metadata.get("response_type", "unknown"),
                    "source": metadata.get("source", "unknown"),
                    "query": None,
                    "answer": None
                }
            
            # Store query or answer text
            if "query" in conv_type:
                conversations[key]["query"] = metadata.get("text", "")
            elif "answer" in conv_type:
                conversations[key]["answer"] = metadata.get("text", "")
        
        # Convert to list and sort by timestamp
        conversation_list = list(conversations.values())
        conversation_list.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return {
            "status": "success",
            "conversations": conversation_list[:limit],
            "total_found": len(conversation_list),
            "user_id": current_user["user_id"],
            "include_answers": include_answers
        }
    
    except Exception as e:
        logger.error(f"Failed to retrieve embedded conversations: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve embedded conversations")

@chat_router.get("/pinecone-stats")
async def get_pinecone_chat_stats(
    current_user: Dict[str, Any] = Depends(require_employee_or_hr)
):
    """Get statistics about embedded chat data in Pinecone"""
    try:
        from db.pinecone_database import get_pinecone_index
        
        idx = get_pinecone_index()
        stats = idx.describe_index_stats()
        
        # Try to get user-specific stats by querying with filters
        dummy_vector = embed_query("test query")
        
        # Count queries for this user
        query_results = idx.query(
            vector=dummy_vector,
            top_k=1,
            include_metadata=True,
            filter={
                "user_id": current_user["user_id"],
                "type": {"$in": ["chat_query", "bulk_test_query"]}
            }
        )
        
        # Count answers for this user
        answer_results = idx.query(
            vector=dummy_vector,
            top_k=1,
            include_metadata=True,
            filter={
                "user_id": current_user["user_id"],
                "type": {"$in": ["chat_answer", "bulk_test_answer"]}
            }
        )
        
        return {
            "status": "success",
            "pinecone_index_stats": {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness
            },
            "user_stats": {
                "user_id": current_user["user_id"],
                "estimated_queries": len(query_results.matches) if query_results else 0,
                "estimated_answers": len(answer_results.matches) if answer_results else 0
            }
        }
    
    except Exception as e:
        logger.error(f"Failed to get Pinecone stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get Pinecone statistics")

# Health check and monitoring
@chat_router.post("/test-patterns")
async def test_pattern_matching(
    current_user: Dict[str, Any] = Depends(require_employee_or_hr)
):
    """Test endpoint to validate pattern matching behavior"""
    test_queries = [
        "working days",
        "working hours", 
        "company name",
        "office hours",
        "what is the company name",
        "tell me working days",
        "ceo name",
        "company address",
        "my leave balance",
        "who is my manager",
        "what department am I in",
        "some random query that should not match"
    ]
    
    try:
        from services.concise_query_matcher import get_concise_query_matcher
        matcher = get_concise_query_matcher()
        
        results = []
        for query in test_queries:
            pattern_result = matcher.process_concise_query(current_user["user_id"], query)
            
            results.append({
                "query": query,
                "matched": pattern_result is not None,
                "pattern": pattern_result.get("pattern") if pattern_result else None,
                "response": pattern_result.get("response") if pattern_result else None,
                "response_length": len(pattern_result.get("response", "")) if pattern_result else 0
            })
        
        return {
            "status": "success",
            "test_results": results,
            "summary": {
                "total_queries": len(test_queries),
                "matched_queries": len([r for r in results if r["matched"]]),
                "unmatched_queries": len([r for r in results if not r["matched"]])
            }
        }
    except Exception as e:
        logger.error(f"Pattern test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pattern test failed: {str(e)}")

@chat_router.post("/analyze-neural-intent")
async def analyze_neural_intent(
    current_user: Dict[str, Any] = Depends(require_employee_or_hr)
):
    """Analyze neural intent classification metrics and uncertainty"""
    test_queries = [
        "working days",
        "my salary", 
        "company name",
        "leave balance",
        "remote work policy",
        "health benefits",
        "ceo name",
        "office hours",
        "my department",
        "mission statement",
        "random unclear query that should be uncertain"
    ]
    
    try:
        from services.enhanced_concise_matcher import get_enhanced_concise_matcher, NEURAL_INTENT_AVAILABLE
        
        if not NEURAL_INTENT_AVAILABLE:
            return {
                "status": "info",
                "message": "Neural intent classification not available. Install torch and scikit-learn, then run training script.",
                "neural_available": False
            }
        
        matcher = get_enhanced_concise_matcher()
        
        if not matcher.neural_classifier or not matcher.neural_classifier.loaded:
            return {
                "status": "info", 
                "message": "Neural intent classifier not loaded. Run: python train_intent_nn.py",
                "neural_available": False
            }
        
        results = []
        
        for query in test_queries:
            # Get neural prediction directly
            neural_result = matcher.neural_classifier.predict_proba(query)
            
            if neural_result:
                category, confidence, prob_map = neural_result
                uncertainty = matcher.neural_classifier.entropy(prob_map)
                
                # Also get rule-based prediction for comparison
                rule_category, rule_action, rule_conf = matcher.intent_detector.detect_intent(query)
                
                results.append({
                    "query": query,
                    "neural": {
                        "category": category,
                        "confidence": round(confidence, 4),
                        "uncertainty": round(uncertainty, 4),
                        "prob_distribution": {k: round(v, 4) for k, v in prob_map.items()}
                    },
                    "rule_based": {
                        "category": rule_category,
                        "action": rule_action, 
                        "confidence": round(rule_conf, 4)
                    },
                    "agreement": category == rule_category
                })
        
        # Calculate summary metrics
        agreements = [r["agreement"] for r in results]
        confidences = [r["neural"]["confidence"] for r in results]
        uncertainties = [r["neural"]["uncertainty"] for r in results]
        
        summary = {
            "neural_rule_agreement": round(sum(agreements) / len(agreements) * 100, 1),
            "average_confidence": round(sum(confidences) / len(confidences), 3),
            "average_uncertainty": round(sum(uncertainties) / len(uncertainties), 3),
            "high_confidence_queries": len([c for c in confidences if c > 0.8]),
            "uncertain_queries": len([u for u in uncertainties if u > 2.0]),
            "model_info": {
                "device": str(matcher.neural_classifier.device),
                "model_type": matcher.neural_classifier.cfg.model_type,
                "num_classes": len(matcher.neural_classifier.labels),
                "classes": matcher.neural_classifier.labels
            }
        }
        
        return {
            "status": "success",
            "neural_available": True,
            "analysis_results": results,
            "summary": summary,
            "interpretation": {
                "confidence": "Higher confidence (closer to 1.0) indicates more certain predictions",
                "uncertainty": "Higher uncertainty (entropy) indicates the model is less sure about the prediction",
                "agreement": "Percentage of queries where neural and rule-based methods agree"
            }
        }
        
    except Exception as e:
        logger.error(f"Neural intent analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Neural intent analysis failed: {str(e)}")

@chat_router.post("/test-enhanced-system")
async def test_enhanced_hrm_system(
    current_user: Dict[str, Any] = Depends(require_employee_or_hr)
):
    """Test the enhanced HRM system with personalization, spelling correction, and LangGraph"""
    test_queries = [
        # Spelling correction tests
        "workng days",
        "companay name", 
        "my salray",
        "ceo nam",
        "ofice ours",
        
        # Personal queries
        "my leave balance",
        "my department",
        "my assets",
        
        # Company queries
        "working hours",
        "mission statement",
        "remote work policy"
    ]
    
    try:
        from services.enhanced_concise_matcher import get_enhanced_concise_matcher
        enhanced_matcher = get_enhanced_concise_matcher()
        
        results = []
        
        for query in test_queries:
            result = enhanced_matcher.process_enhanced_query(current_user["user_id"], query)
            
            test_result = {
                "query": query,
                "success": result is not None and result.get("success", False),
                "response": result.get("response") if result else None,
                "pattern": result.get("pattern") if result else None,
                "metadata": result.get("metadata", {}) if result else {}
            }
            
            if result:
                metadata = result.get("metadata", {})
                response = result.get("response", "")
                
                # Analyze response quality
                test_result["analysis"] = {
                    "original_query": metadata.get("original_query", query),
                    "corrected_query": metadata.get("corrected_query", query),
                    "spelling_corrected": metadata.get("original_query") != metadata.get("corrected_query"),
                    "intent_category": metadata.get("intent_category"),
                    "intent_action": metadata.get("intent_action"),
                    "personalized": response.startswith("Hi "),
                    "has_follow_up": "?" in response and len(response.split("?")) > 1,
                    "response_length": len(response),
                    "is_concise": len(response) < 300,
                    "processing_steps": metadata.get("processing_steps", [])
                }
            
            results.append(test_result)
        
        # Calculate summary statistics
        successful_queries = [r for r in results if r["success"]]
        personalized_responses = [r for r in results if r.get("analysis", {}).get("personalized", False)]
        spelling_corrected = [r for r in results if r.get("analysis", {}).get("spelling_corrected", False)]
        with_followup = [r for r in results if r.get("analysis", {}).get("has_follow_up", False)]
        concise_responses = [r for r in results if r.get("analysis", {}).get("is_concise", False)]
        
        summary = {
            "total_queries": len(test_queries),
            "successful_responses": len(successful_queries),
            "success_rate": round(len(successful_queries) / len(test_queries) * 100, 1),
            "personalized_responses": len(personalized_responses),
            "spelling_corrections": len(spelling_corrected),
            "follow_up_prompts": len(with_followup),
            "concise_responses": len(concise_responses),
            "average_response_length": round(sum(r.get("analysis", {}).get("response_length", 0) for r in results) / len(results), 1) if results else 0
        }
        
        return {
            "status": "success",
            "test_results": results,
            "summary": summary,
            "features_tested": [
                "Spelling correction",
                "Intent detection", 
                "Pattern matching",
                "Personalization",
                "Follow-up prompts",
                "LangGraph workflow",
                "Direct database queries"
            ]
        }
        
    except Exception as e:
        logger.error(f"Enhanced system test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced system test failed: {str(e)}")


@chat_router.get("/health")
async def health_check():
    """Enhanced health check with component status"""
    try:
        db = get_database()
        
        # Test database connection
        db_status = "healthy"
        try:
            db.command("ping")
        except:
            db_status = "unhealthy"
        
        # Test Pinecone connection
        pinecone_status = "healthy"
        try:
            idx = get_pinecone_index()
            idx.describe_index_stats()
        except:
            pinecone_status = "unhealthy"
        
        # Test embedding model
        embedding_status = "healthy"
        try:
            embed_query("test")
        except:
            embedding_status = "unhealthy"
        
        return {
            "status": "healthy" if all(s == "healthy" for s in [db_status, pinecone_status, embedding_status]) else "degraded",
            "components": {
                "database": db_status,
                "pinecone": pinecone_status,
                "embeddings": embedding_status
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }



def flatten_json(obj):
    if isinstance(obj, dict):
        return " ".join([f"{k}: {flatten_json(v)}" for k, v in obj.items()])
    elif isinstance(obj, list):
        return " ".join([flatten_json(x) for x in obj])
    return str(obj)

def generate_text_from_item(item):
    coll = item["collection"]
    data = item["data"]

    if coll == "company_info":
        return (
            f"Company {data.get('name')} founded in {data.get('founded_year')} by {', '.join(data.get('founded_by', []))}. "
            f"CEO: {data.get('ceo_name')}. Mission: {data.get('mission_statement')}. "
            f"Vision: {data.get('vision_statement')}. Core Values: {', '.join(data.get('core_values', []))}. "
            f"Headquarters: {data.get('headquarters_address')}."
        )
    if coll == "employees":
        return (
            f"Employee {data.get('first_name')} {data.get('last_name')} works in {data.get('department_id')} as {data.get('designation_id')}. "
            f"Email: {data.get('email')}, Location: {data.get('location')}, Salary: {data.get('base_salary')} {data.get('salary_currency')}, Grade: {data.get('grade')}."
        )
    if coll == "leave_balances":
        return (
            f"User {data.get('user_id')} has {data.get('balance')} days of {data.get('leave_type_code')} leave as of {data.get('as_of')}."
        )
    if coll == "payroll_records":
        return (
            f"Payroll for User {data.get('user_id')} for {data.get('month')}/{data.get('year')}: "
            f"Gross Pay {data.get('gross_pay')} {data.get('currency')}, Net Pay {data.get('net_pay')} {data.get('currency')}. "
            f"Components: {data.get('components')}, Deductions: {data.get('deductions')}."
        )
    if coll == "designations":
        return (
            f"Designation {data.get('title')} in Department {data.get('department_id')} - {data.get('description')}."
        )
    if coll == "departments":
        return (
            f"Department {data.get('name')} - {data.get('description')}."
        )
    return flatten_json(data)

@chat_router.post("/train-custom-model")
def prepare_and_store_embeddings(json_file_path: str) -> dict:
    with open(json_file_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    docs = []

    for i, item in enumerate(raw):
        text = generate_text_from_item(item)
        vector = model.encode(text).tolist()

        insert_vector(
            id=str(i),
            vector=vector,
            metadata={
                "text": text,
                "collection": item["collection"],
                "original_json": json.dumps(item)
            }
        )

        docs.append(text)

    return {"status": "success", "chunks_prepared": len(docs), "chunks_inserted": len(docs)}

@chat_router.get("/get-answer")
def answer_question(query: str, top_k: int = 4) -> dict:
    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    q_emb = model.encode(query).tolist()  # Ensure it's a list

    # Query Pinecone or your vector DB
    matches = query_vector(q_emb, top_k=top_k)
    results = []
    answers = []

    for match in matches:
        meta = match.metadata if hasattr(match, "metadata") else match.get("metadata", {})
        original_json = meta.get("original_json")
        best_answer = None
        best_score = -1

        if original_json:
            data_obj = json.loads(original_json)
            data = data_obj.get("data", {})
            collection = data_obj.get("collection")

            # Compare query against each field semantically
            for key, value in data.items():
                if value is None:
                    continue
                field_text = f"{key}: {value}"
                field_emb = model.encode(field_text).tolist()
                score = cosine_similarity([q_emb], [field_emb])[0][0]

                if score > best_score:
                    best_score = score
                    best_answer = field_text

        answers.append(best_answer)

        results.append({
            "vector_id": match.id if hasattr(match, "id") else match.get("id"),
            "score": match.score if hasattr(match, "score") else match.get("score"),
            "collection": meta.get("collection"),
            "answer": best_answer
        })

    return {
        "query": query,
        "top_k": top_k,
        "answers": answers,
        "results": results
    }
def chunk_text(text, max_chars=800):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

@chat_router.post("/train-story-embedding")
def train_story_embedding(story_text: str) -> dict:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunks = chunk_text(story_text)

    for i, chunk in enumerate(chunks):
        vector = model.encode(chunk).tolist()
        insert_vector(
            id=f"story_chunk_{i}",
            vector=vector,
            metadata={"text": chunk, "type": "story"}
        )

    return {"status": "success", "chunks_stored": len(chunks)}

@chat_router.get("/search-story")
def search_story(query: str, top_k: int = 3) -> dict:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from nltk.tokenize import sent_tokenize
    from rapidfuzz import fuzz

    model = SentenceTransformer("all-MiniLM-L6-v2")
    q_emb = model.encode(query).tolist()
    matches = query_vector(q_emb, top_k=top_k)

    results = []

    for match in matches:
        meta = match.metadata if hasattr(match, "metadata") else match.get("metadata", {})
        chunk_text = meta.get("text", "")
        best_sentence = None
        best_score = -1

        sentences = sent_tokenize(chunk_text)
        for sentence in sentences:
            if not sentence.strip():
                continue
            s_emb = model.encode(sentence).tolist()
            semantic_score = cosine_similarity([q_emb], [s_emb])[0][0]
            fuzzy_score = fuzz.partial_ratio(query.lower(), sentence.lower()) / 100.0
            combined_score = 0.7 * semantic_score + 0.3 * fuzzy_score

            if combined_score > best_score:
                best_score = combined_score
                best_sentence = sentence.strip()

        results.append({
            "chunk_id": match.id,
            "score": match.score,
            "best_sentence": best_sentence,
            "similarity": best_score
        })

    return {
        "query": query,
        "results": results
    }
def serialize_mongo_doc(doc):
    if isinstance(doc, list):
        return [serialize_mongo_doc(d) for d in doc]
    if isinstance(doc, dict):
        new_doc = {}
        for k, v in doc.items():
            if isinstance(v, ObjectId):
                new_doc[k] = str(v)
            elif isinstance(v, (dict, list)):
                new_doc[k] = serialize_mongo_doc(v)
            else:
                new_doc[k] = v
        return new_doc
    return doc


@chat_router.post("/history")
async def get_chat_history(
    current_user: Dict[str, Any] = Depends(require_employee_or_hr)
):
    try:
        db = get_database()
        history = list(db["chat_sessions"].find({"user_id": current_user["user_id"]}))
        serialized = serialize_mongo_doc(history)

        return {
            "status": "success",
            "status_code": status.HTTP_200_OK,
            "data": serialized
        }

    except PyMongoError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


@chat_router.post("/history/delete")
async def delete_chat_history(
    delete_chat_request: DeleteChatRequest,
    current_user: Dict[str, Any] = Depends(require_employee_or_hr)
):
    try:
        db = get_database()
        result = db["chat_sessions"].delete_one({
            "session_id": delete_chat_request.session_id,
            "user_id": current_user["user_id"]
        })

        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found or already deleted"
            )

        return {
            "status": "success",
            "status_code": status.HTTP_200_OK,
            "data": {
                "message": "Chat history deleted successfully",
                "deleted_session_id": delete_chat_request.session_id
            }
        }

    except PyMongoError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


@chat_router.post("/history/clear")
async def delete_all_chat_history(
    current_user: Dict[str, Any] = Depends(require_employee_or_hr)
):
    try:
        db = get_database()
        db["chat_messages"].delete_many({"user_id": current_user["user_id"]})
        return {
            "status": "success",
            "status_code": status.HTTP_200_OK,
            "data": {
                "message": "Chat history deleted successfully"
            }
        }
    except PyMongoError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )

@chat_router.post("/history/delete-all-sessions")
async def delete_all_chat_sessions(
    current_user: Dict[str, Any] = Depends(require_employee_or_hr)
):
    try:
        db = get_database()
        db["chat_sessions"].delete_many({"user_id": current_user["user_id"]})
        return {"status": "success"}
    except PyMongoError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"    
        )

@chat_router.post("/history/{session_id}")
async def get_chat_history_by_session_id(
    session_id: str,
    current_user: Dict[str, Any] = Depends(require_employee_or_hr)
):
    try:
        db = get_database()
        messages = list(db["chat_messages"].find({
            "session_id": session_id,
            "user_id": current_user["user_id"]
        }))

        if not messages:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No chat messages found for this session"
            )

        serialized = serialize_mongo_doc(messages)

        return {
            "status": "success",
            "status_code": status.HTTP_200_OK,
            "data": serialized
        }

    except PyMongoError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


    """
    Create a new chat session for the logged-in user.
    """
    try:
        db = get_database()

        # Build the session document
        new_session = {
            "user_id": current_user["user_id"],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "title": session_data.get("title", "New Chat Session"),
            "messages": session_data.get("messages", []),
            "status": "active"
        }

        # Insert into MongoDB
        result = db["chat_sessions"].insert_one(new_session)
        created_session = db["chat_sessions"].find_one({"_id": result.inserted_id})
        serialized = serialize_mongo_doc(created_session)

        return {
            "status": "success",
            "status_code": status.HTTP_201_CREATED,
            "data": serialized
        }

    except PyMongoError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )