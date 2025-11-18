# enhanced_chat.py
import uuid

from typing import Dict, Any
from datetime import datetime

from bson import ObjectId

from services.mistralai import get_mistral_with_context_service
from fastapi import APIRouter, HTTPException,Body,Depends,status

from pymongo.errors import PyMongoError

from core.middleware import require_employee_or_hr
from config.database import get_database
from models.index import ChatRequest, ChatResponse, KnowledgeDocument, DeleteChatRequest
from config.pinecone import insert_vector
from core.graph_agent import  embed_query
# from services.hrm_agent_service import get_hrm_agent_service
from services.get_user_data import get_user_hr_data
from config.logger import logger
# Pinecone metadata only supports strings, numbers, booleans, or lists of strings.
from helpers._sanitize_pinecone_metadata import _sanitize_pinecone_metadata
from helpers.serialize_mongo_doc import serialize_mongo_doc
chat_router = APIRouter(prefix="/chat", tags=["Enhanced AI Chat"])

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



@chat_router.post("/history")
async def get_chat_history(
    current_user: Dict[str, Any] = Depends(require_employee_or_hr)
):
    try:
        db = get_database()
        history = list[Any](db["chat_sessions"].find({"user_id": current_user["user_id"]}))
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