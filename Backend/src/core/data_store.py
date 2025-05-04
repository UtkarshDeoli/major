from typing import Dict, List, Any, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from bson import ObjectId
from datetime import datetime
import os
import json

# Import settings from config
from src.core.config import MONGODB_URL, MONGODB_DB_NAME, MONGODB_CONNECT_TIMEOUT

# MongoDB connection with error handling
try:
    client = AsyncIOMotorClient(
        MONGODB_URL,
        serverSelectionTimeoutMS=MONGODB_CONNECT_TIMEOUT
    )
    # Force a connection to verify it works
    client.admin.command('ismaster')
    print(f"Connected to MongoDB at {MONGODB_URL}")
    
    db = client[MONGODB_DB_NAME]
    # Collections
    users_collection = db.users
    pdfs_collection = db.pdfs
    chat_sessions_collection = db.chat_sessions
except (ConnectionFailure, ServerSelectionTimeoutError) as e:
    print(f"MongoDB connection error: {e}")
    print("WARNING: Data store service will not work until MongoDB is available")
    # We'll initialize these as None and check before each operation
    client = None
    db = None
    users_collection = None
    pdfs_collection = None
    chat_sessions_collection = None

# Helper to convert ObjectId to string
def object_id_to_str(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, ObjectId):
                obj[k] = str(v)
            elif isinstance(v, (dict, list)):
                object_id_to_str(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, ObjectId):
                obj[i] = str(v)
            elif isinstance(v, (dict, list)):
                object_id_to_str(v)
    return obj

# PDF operations
async def store_pdf_metadata(
    filename: str,
    size: int,
    user_id: str,
    file_path: str,
    processed: bool = False,
    title: Optional[str] = None,
    description: Optional[str] = None,
    page_count: Optional[int] = None,
    vector_db_path: Optional[str] = None,
    tags: Optional[List[str]] = None
):
    """Store PDF metadata in MongoDB"""
    if pdfs_collection is None:
        raise Exception("Database connection not available")
    
    pdf_data = {
        "filename": filename,
        "size": size,
        "upload_date": datetime.now(),
        "user_id": user_id,
        "file_path": file_path,
        "processed": processed,
        "title": title or filename,
        "description": description,
        "page_count": page_count,
        "vector_db_path": vector_db_path,
        "tags": tags or []
    }
    
    result = await pdfs_collection.insert_one(pdf_data)
    pdf_data["id"] = str(result.inserted_id)
    return object_id_to_str(pdf_data)

async def update_pdf_metadata(pdf_id: str, update_data: Dict[str, Any]):
    """Update PDF metadata in MongoDB"""
    if pdfs_collection is None:
        raise Exception("Database connection not available")
    
    update_data["updated_at"] = datetime.now()
    
    await pdfs_collection.update_one(
        {"_id": ObjectId(pdf_id)},
        {"$set": update_data}
    )
    
    updated_pdf = await pdfs_collection.find_one({"_id": ObjectId(pdf_id)})
    if updated_pdf:
        updated_pdf["id"] = str(updated_pdf["_id"])
        del updated_pdf["_id"]
        return updated_pdf
    return None

async def get_user_pdfs(user_id: str):
    """Get all PDFs uploaded by a specific user"""
    if pdfs_collection is None:
        raise Exception("Database connection not available")
    
    cursor = pdfs_collection.find({"user_id": user_id})
    pdf_list = []
    
    async for pdf in cursor:
        pdf["id"] = str(pdf["_id"])
        del pdf["_id"]
        pdf_list.append(pdf)
    
    return pdf_list

async def get_pdf_metadata(pdf_id: str):
    """Get metadata for a specific PDF"""
    if pdfs_collection is None:
        raise Exception("Database connection not available")
    
    pdf = await pdfs_collection.find_one({"_id": ObjectId(pdf_id)})
    if pdf:
        pdf["id"] = str(pdf["_id"])
        del pdf["_id"]
        return pdf
    return None

# Chat history operations
async def create_chat_session(user_id: str, title: str, pdf_id: Optional[str] = None):
    """Create a new chat session"""
    if chat_sessions_collection is None:
        raise Exception("Database connection not available")
    
    now = datetime.now()
    chat_data = {
        "user_id": user_id,
        "pdf_id": pdf_id,
        "title": title,
        "messages": [],
        "created_at": now,
        "updated_at": now
    }
    
    result = await chat_sessions_collection.insert_one(chat_data)
    chat_data["id"] = str(result.inserted_id)
    return object_id_to_str(chat_data)

async def get_user_chat_sessions(user_id: str):
    """Get all chat sessions for a user"""
    if chat_sessions_collection is None:
        raise Exception("Database connection not available")
    
    cursor = chat_sessions_collection.find({"user_id": user_id})
    chat_sessions = []
    
    async for session in cursor:
        session["id"] = str(session["_id"])
        del session["_id"]
        # Add message count
        session["message_count"] = len(session.get("messages", []))
        chat_sessions.append(session)
    
    return chat_sessions

async def get_chat_session(session_id: str):
    """Get a specific chat session with all messages"""
    if chat_sessions_collection is None:
        raise Exception("Database connection not available")
    
    session = await chat_sessions_collection.find_one({"_id": ObjectId(session_id)})
    if session:
        session["id"] = str(session["_id"])
        del session["_id"]
        return object_id_to_str(session)
    return None

async def add_message_to_chat(session_id: str, role: str, content: str):
    """Add a message to a chat session"""
    if chat_sessions_collection is None:
        raise Exception("Database connection not available")
    
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now()
    }
    
    await chat_sessions_collection.update_one(
        {"_id": ObjectId(session_id)},
        {
            "$push": {"messages": message},
            "$set": {"updated_at": datetime.now()}
        }
    )
    
    # Return the message with an ID
    message["id"] = str(ObjectId())
    return message

async def save_vector_db(pdf_id: str, vector_data: Dict[str, Any]):
    """Save vector database to a JSON file"""
    user_pdf = await get_pdf_metadata(pdf_id)
    if not user_pdf:
        raise Exception(f"PDF with ID {pdf_id} not found")
    
    # Create processed directory if it doesn't exist
    processed_dir = os.path.join("processed", user_pdf["user_id"])
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save vector database to a JSON file
    vector_db_path = os.path.join(processed_dir, f"{pdf_id}.json")
    with open(vector_db_path, "w") as f:
        json.dump(vector_data, f)
    
    # Update PDF metadata with vector database path
    await update_pdf_metadata(pdf_id, {
        "processed": True,
        "vector_db_path": vector_db_path
    })
    
    return vector_db_path

async def load_vector_db(pdf_id: str):
    """Load vector database from a JSON file"""
    user_pdf = await get_pdf_metadata(pdf_id)
    if not user_pdf or not user_pdf.get("vector_db_path"):
        raise Exception(f"Vector database for PDF ID {pdf_id} not found")
    
    # Load vector database from JSON file
    with open(user_pdf["vector_db_path"], "r") as f:
        vector_data = json.load(f)
    
    return vector_data
