from typing import Dict, List, Any, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from bson import ObjectId
from datetime import datetime
import os
import json
import asyncio

# Import settings from config
from src.core.config import MONGODB_URL, MONGODB_DB_NAME, MONGODB_CONNECT_TIMEOUT
# MongoDB connection with error handling
try:
    client = AsyncIOMotorClient(
        MONGODB_URL,
        serverSelectionTimeoutMS=MONGODB_CONNECT_TIMEOUT
    )
    # Force a connection to verify it works (run coroutine on current event loop if possible)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Can't run loop from here; skip immediate check and rely on Motor's lazy connection.
            print("Event loop is already running; skipping initial MongoDB connection check")
        else:
            loop.run_until_complete(client.admin.command('ismaster'))
            print(f"Connected to MongoDB at {MONGODB_URL}")
    except Exception:
        # If the synchronous check fails, we'll rely on lazy connection and let actual DB ops surface issues.
        print(f"Could not verify MongoDB connection at import time; will attempt on first use.")
    
    db = client[MONGODB_DB_NAME]
    # Collections
    users_collection = db.users
    pdfs_collection = db.pdfs
    materials_collection = db.materials
    chat_sessions_collection = db.chat_sessions
    # Mock Test Collections
    mock_tests_collection = db.mock_tests
    mock_test_submissions_collection = db.mock_test_submissions
    # Workspace Collections
    exams_collection = db.exams
    subjects_collection = db.subjects
    collections_collection = db.collections
    document_chunks_collection = db.document_chunks
except (ConnectionFailure, ServerSelectionTimeoutError) as e:
    print(f"MongoDB connection error: {e}")
    print("WARNING: Data store service will not work until MongoDB is available")
    # We'll initialize these as None and check before each operation
    client = None
    db = None
    users_collection = None
    pdfs_collection = None
    materials_collection = None
    chat_sessions_collection = None
    # Mock Test Collections
    mock_tests_collection = None
    mock_test_submissions_collection = None
    # Workspace Collections
    exams_collection = None
    subjects_collection = None
    collections_collection = None
    document_chunks_collection = None


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
async def create_chat_session(user_id: str, title: str, pdf_id: Optional[str] = None, doc_ids: Optional[List[str]] = None):
    """Create a new chat session"""
    if chat_sessions_collection is None:
        raise Exception("Database connection not available")

    now = datetime.now()
    chat_data = {
        "user_id": user_id,
        "pdf_id": pdf_id,
        "doc_ids": doc_ids,
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

# Mock Test Functions
async def store_mock_test(mock_test_data: Dict[str, Any]) -> str:
    """Store a mock test in the database"""
    if mock_tests_collection is None:
        raise Exception("Database connection not available")
    
    try:
        result = await mock_tests_collection.insert_one(mock_test_data)
        return str(result.inserted_id)
    except Exception as e:
        raise Exception(f"Error storing mock test: {str(e)}")

async def get_user_mock_tests(user_id: str) -> List[Dict[str, Any]]:
    """Get all mock tests for a user"""
    if mock_tests_collection is None:
        raise Exception("Database connection not available")
    
    try:
        cursor = mock_tests_collection.find({
            "$or": [
                {"user_id": user_id},
                {"assigned_to": user_id},
                {"created_by": user_id}
            ]
        }).sort("created_at", -1)
        tests = await cursor.to_list(length=None)
        return [object_id_to_str(test) for test in tests]
    except Exception as e:
        raise Exception(f"Error fetching user mock tests: {str(e)}")

async def get_mock_test(test_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific mock test by ID"""
    if mock_tests_collection is None:
        raise Exception("Database connection not available")
    
    try:
        test = await mock_tests_collection.find_one({"test_id": test_id})
        return object_id_to_str(test) if test else None
    except Exception as e:
        raise Exception(f"Error fetching mock test: {str(e)}")

async def store_mock_test_submission(submission_data: Dict[str, Any]) -> str:
    """Store a mock test submission in the database"""
    if mock_test_submissions_collection is None:
        raise Exception("Database connection not available")
    
    try:
        result = await mock_test_submissions_collection.insert_one(submission_data)
        return str(result.inserted_id)
    except Exception as e:
        raise Exception(f"Error storing mock test submission: {str(e)}")

async def get_user_mock_test_submissions(user_id: str) -> List[Dict[str, Any]]:
    """Get all mock test submissions for a user"""
    if mock_test_submissions_collection is None:
        raise Exception("Database connection not available")

    try:
        cursor = mock_test_submissions_collection.find({"user_id": user_id}).sort("created_at", -1)
        submissions = await cursor.to_list(length=None)
        return [object_id_to_str(submission) for submission in submissions]
    except Exception as e:
        raise Exception(f"Error fetching user mock test submissions: {str(e)}")

# Document chunk operations
async def store_document_chunks(chunks: List[Dict[str, Any]]):
    """Store document chunks in MongoDB."""
    if document_chunks_collection is None:
        raise Exception("Database connection not available")

    if not chunks:
        return []

    result = await document_chunks_collection.insert_many(chunks)
    return [str(id) for id in result.inserted_ids]

async def get_chunks_by_chroma_ids(chroma_ids: List[str]):
    """Fetch chunks by their ChromaDB IDs."""
    if document_chunks_collection is None:
        raise Exception("Database connection not available")

    cursor = document_chunks_collection.find({"chroma_id": {"$in": chroma_ids}})
    chunks = await cursor.to_list(length=None)

    chunk_map = {c["chroma_id"]: c for c in chunks}
    ordered = [chunk_map.get(cid) for cid in chroma_ids if cid in chunk_map]

    return [c for c in ordered if c]

async def get_user_chunks_for_bm25(user_id: str):
    """Get all chunks for a user to build BM25 index."""
    if document_chunks_collection is None:
        raise Exception("Database connection not available")

    cursor = document_chunks_collection.find({"user_id": user_id})
    return await cursor.to_list(length=None)

async def delete_document_chunks(doc_id: str):
    """Delete all chunks for a document."""
    if document_chunks_collection is None:
        raise Exception("Database connection not available")

    await document_chunks_collection.delete_many({"doc_id": doc_id})

async def update_chunk_tags(doc_id: str, tags: List[str]):
    """Update tags for all chunks of a document."""
    if document_chunks_collection is None:
        raise Exception("Database connection not available")

    await document_chunks_collection.update_many(
        {"doc_id": doc_id},
        {"$set": {"tags": tags}}
    )

async def ensure_indexes():
    """Create indexes for document_chunks collection."""
    if document_chunks_collection is None:
        return

    await document_chunks_collection.create_index([("user_id", 1), ("doc_id", 1)])
    await document_chunks_collection.create_index([("user_id", 1), ("subject", 1)])
    await document_chunks_collection.create_index([("user_id", 1), ("tags", 1)])
    await document_chunks_collection.create_index("chroma_id", unique=True)
