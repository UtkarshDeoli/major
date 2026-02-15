--
Orbit Backend - FastAPI + Python
Overview
AI-powered study platform backend built with FastAPI, PostgreSQL, and OpenRouter (free models).
Tech Stack
| Layer | Technology |
|-------|------------|
| Web Framework | FastAPI + Uvicorn |
| Database | PostgreSQL + SQLAlchemy |
| Authentication | JWT (python-jose) |
| AI Integration | OpenRouter API (free models) |
| File Storage | Local filesystem / S3 |
| Vector Database | Qdrant (for embeddings) |
Free Models Configuration
FREE_MODELS = {
    "chat": "meta-llama/llama-3.3-70b-instruct",      # 128K context
    "quiz": "deepseek/deepseek-r1-0528",                # 164K context
    "analysis": "stepfun/step-3.5-flash",               # 256K context
    "fallback": "openrouter/free",                       # Auto-router
}
Project Structure
orbit-backend/
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI app entry
│   ├── config.py                   # Settings, environment variables
│   ├── database.py                 # Database connection
│   ├── models/                     # SQLAlchemy models
│   │   ├── __init__.py
│   │   ├── user.py
│   │   ├── pdf.py
│   │   ├── chat_session.py
│   │   └── mock_test.py
│   ├── schemas/                    # Pydantic schemas
│   │   ├── __init__.py
│   │   ├── user.py
│   │   ├── pdf.py
│   │   ├── chat.py
│   │   └── mock_test.py
│   ├── routers/                    # API endpoints
│   │   ├── __init__.py
│   │   ├── auth.py                 # POST /auth/login, /auth/signup
│   │   ├── pdfs.py                 # POST /pdfs/upload, GET /pdfs/
│   │   ├── chat.py                 # POST /questions/ask, /questions/sessions
│   │   ├── analysis.py             # POST /analysis/question-papers
│   │   └── mock_tests.py            # POST /mock-tests/generate
│   ├── services/                    # Business logic
│   │   ├── __init__.py
│   │   ├── ai_service.py           # OpenRouter integration
│   │   ├── pdf_service.py          # PDF processing
│   │   ├── embedding_service.py    # Vector embeddings
│   │   └── quiz_service.py         # Quiz generation
│   └── dependencies/                # Auth, database, etc.
│       ├── __init__.py
│       ├── auth.py                 # JWT dependency
│       └── database.py             # DB session
├── uploads/                        # Uploaded PDF files
├── tests/                           # Unit tests
├── alembic/                         # Database migrations
├── .env                            # Environment variables
├── requirements.txt
├── Dockerfile
└── README.md
API Endpoints
Authentication (/auth)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /auth/login | Login (OAuth2 form) |
| POST | /auth/signup | Register user |
| POST | /auth/logout | Logout |
| GET | /auth/me | Get current user |
PDF Management (/pdfs)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /pdfs/upload | Upload PDF (multipart) |
| GET | /pdfs/ | List user PDFs |
| GET | /pdfs/{id} | Get PDF metadata |
| GET | /pdfs/{id}/download | Download PDF |
| DELETE | /pdfs/{id} | Delete PDF |
Chat (/questions)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /questions/ask | Ask question (no history) |
| POST | /questions/ask/stream | Stream response (SSE) |
| POST | /questions/sessions | Create chat session |
| GET | /questions/sessions/ | List sessions |
| GET | /questions/sessions/{id} | Get session + messages |
| POST | /questions/sessions/{id}/messages | Add message + get response |
| DELETE | /questions/sessions/{id} | Delete session |
Analysis (/analysis)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /analysis/question-papers | Analyze syllabus + papers |
Mock Tests (/mock-tests)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /mock-tests/generate | Generate mock test |
| GET | /mock-tests/ | List user tests |
| GET | /mock-tests/{id} | Get test with questions |
| POST | /mock-tests/{id}/submit | Submit answers |
| GET | /mock-tests/submissions/{id}/analysis | Get results |
Database Schema
-- Users
CREATE TABLE users (
    id UUID PRIMARY KEY,
    email VARCHAR UNIQUE NOT NULL,
    hashed_password VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
-- PDFs
CREATE TABLE pdfs (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    title VARCHAR NOT NULL,
    description TEXT,
    tags JSONB,
    file_path VARCHAR NOT NULL,
    file_size INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
-- Chat Sessions
CREATE TABLE chat_sessions (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    pdf_id UUID REFERENCES pdfs(id),
    title VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
-- Chat Messages
CREATE TABLE chat_messages (
    id UUID PRIMARY KEY,
    session_id UUID REFERENCES chat_sessions(id),
    role VARCHAR NOT NULL,  -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
-- Mock Tests
CREATE TABLE mock_tests (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    title VARCHAR NOT NULL,
    questions JSONB NOT NULL,
    total_marks INTEGER,
    difficulty VARCHAR,
    time_limit INTEGER,  -- minutes
    created_at TIMESTAMP DEFAULT NOW()
);
-- Test Submissions
CREATE TABLE test_submissions (
    id UUID PRIMARY KEY,
    test_id UUID REFERENCES mock_tests(id),
    answers JSONB NOT NULL,
    time_taken INTEGER,  -- seconds
    score DECIMAL,
    analysis JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
Environment Variables
# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/orbit
# JWT
SECRET_KEY=your-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
# OpenRouter
OPENROUTER_API_KEY=sk-or-...
# File Storage
UPLOAD_DIR=./uploads
MAX_FILE_SIZE=10485760  # 10MB
# App
DEBUG=true
CORS_ORIGINS=http://localhost:3000
Installation
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
# Install dependencies
pip install -r requirements.txt
# Set up database
alembic upgrade head
# Run development server
uvicorn app.main:app --reload
Requirements.txt
fastapi==0.109.0
uvicorn[standard]==0.27.0
sqlalchemy[asyncio]==2.0.25
asyncpg==0.29.0
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6
openai==1.12.0
qdrant-client==1.7.0
python-dotenv==1.0.0
pydantic==2.5.3
pydantic-settings==2.1.0
pypdf==3.17.0
aiofiles==23.2.1
AI Service Integration
# app/services/ai_service.py
from openai import AsyncOpenAI
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
FREE_MODELS = {
    "chat": "meta-llama/llama-3.3-70b-instruct",
    "quiz": "deepseek/deepseek-r1-0528",
    "analysis": "stepfun/step-3.5-flash",
}
async def get_ai_response(prompt: str, model_type: str = "chat"):
    model = FREE_MODELS.get(model_type, FREE_MODELS["fallback"])
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content
Deployment (Docker)
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
Frontend Integration
Base URL: http://localhost:8000
// Frontend API client (Next.js)
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
// Auth headers
const getAuthHeaders = () => ({
  Authorization: `Bearer ${localStorage.getItem('token')}`,
});
// Upload PDF
const uploadPDF = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  return fetch(`${API_BASE_URL}/pdfs/upload`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}` },
    body: formData,
  });
};
// Chat with AI
const askQuestion = async (question: string, pdfId?: string) => {
  return fetch(`${API_BASE_URL}/questions/ask`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ question, pdf_id: pdfId }),
  });
};
Features Implemented
1. User Authentication - JWT-based login/signup
2. PDF Upload & Storage - Local filesystem or S3
3. AI Chat - Conversational Q&A with document context
4. Smart Search - Vector similarity search with Qdrant
5. Quiz Generation - AI-generated MCQ + descriptive questions
6. Test Analysis - Performance insights and recommendations
7. Progress Tracking - User stats and quiz history
Next Steps
1. Set up PostgreSQL database
2. Configure environment variables
3. Run database migrations
4. Implement auth endpoints
5. Build PDF upload service
6. Integrate OpenRouter AI
7. Create all API routers
8. Write unit tests
9. Dockerize for deployment
---
Copy this content to README-backend.md and pass it to OpenCode to implement the backend!
▣  Plan · minimax-m2.1-free · 58.8s
also create a todo that i can put in todo.md for llm to chek 
Thinking: The user is asking me to create a TODO file, but I'm in Plan Mode (read-only), so I cannot create files. I need to provide the content for the TODO.md file that the user can copy and save themselves.
Let me provide a comprehensive todo list for implementing the Orbit backend.
I'm in Plan Mode (read-only), so I cannot create files. Here's the content for your TODO.md:
---
Orbit Backend - Implementation TODO
Phase 1: Project Setup
- [ ] Initialize Python project with requirements.txt
- [ ] Create FastAPI project structure (app/ directory)
- [ ] Set up config.py with environment variables
- [ ] Create database.py with PostgreSQL connection
- [ ] Configure CORS for frontend (localhost:3000)
Phase 2: Database Models
- [ ] Create models/user.py - User table with email/password
- [ ] Create models/pdf.py - PDF metadata storage
- [ ] Create models/chat_session.py - Chat sessions
- [ ] Create models/chat_message.py - Chat messages
- [ ] Create models/mock_test.py - Generated tests
- [ ] Create models/test_submission.py - Test submissions
- [ ] Run Alembic migrations
Phase 3: Authentication
- [ ] Implement password hashing (bcrypt)
- [ ] Create JWT token generation
- [ ] Build /auth/signup endpoint
- [ ] Build /auth/login endpoint
- [ ] Build /auth/me endpoint with JWT dependency
- [ ] Add OAuth2 form handling
Phase 4: PDF Management
- [ ] Create file upload handler (multipart/form-data)
- [ ] Build /pdfs/upload endpoint
- [ ] Build /pdfs/ - List user PDFs
- [ ] Build /pdfs/{id} - Get PDF details
- [ ] Build /pdfs/{id}/download - Download file
- [ ] Build /pdfs/{id} - Delete PDF
- [ ] Set up file storage (local/uploads or S3)
Phase 5: AI Integration (OpenRouter)
- [ ] Install openai package
- [ ] Configure OpenRouter client with free models
- [ ] Create ai_service.py:
  - [ ] Chat completion function
  - [ ] Streaming response handler (SSE)
  - [ ] Quiz generation prompt
  - [ ] Analysis prompt templates
Phase 6: Chat System
- [ ] Build /questions/ask - One-off Q&A
- [ ] Build /questions/ask/stream - SSE streaming
- [ ] Build /questions/sessions - Create session
- [ ] Build /questions/sessions/ - List sessions
- [ ] Build /questions/sessions/{id} - Get session
- [ ] Build /questions/sessions/{id}/messages - Add message + AI response
- [ ] Build /questions/sessions/{id} - Delete session
Phase 7: PDF Text Extraction
- [ ] Install pypdf or pdfplumber
- [ ] Create pdf_service.py:
  - [ ] Extract text from PDF
  - [ ] Chunk text for embeddings
  - [ ] Store embeddings in Qdrant (optional)
Phase 8: Analysis Endpoint
- [ ] Build /analysis/question-papers endpoint
- [ ] Extract text from syllabus + question papers
- [ ] Send to AI for pattern analysis
- [ ] Return structured analysis:
  - [ ] Overall summary
  - [ ] Focus areas
  - [ ] Unit-wise analysis
  - [ ] Question patterns
  - [ ] Sample questions
  - [ ] Preparation strategy
Phase 9: Quiz/Mock Test Generation
- [ ] Build /mock-tests/generate endpoint
- [ ] Accept: syllabus, question papers, config
- [ ] Generate MCQ questions
- [ ] Generate descriptive questions
- [ ] Store test in database
- [ ] Return test ID + questions
Phase 10: Test Taking & Results
- [ ] Build /mock-tests/{id} - Get test questions
- [ ] Build /mock-tests/{id}/submit - Submit answers
- [ ] AI grading of answers
- [ ] Calculate score + percentage
- [ ] Generate performance analysis
- [ ] Build /mock-tests/submissions/{id}/analysis - Get results
Phase 11: Testing
- [ ] Write unit tests for auth
- [ ] Write unit tests for PDF endpoints
- [ ] Write unit tests for chat
- [ ] Write unit tests for AI service
- [ ] Write integration tests
Phase 12: Docker & Deployment
- [ ] Create Dockerfile
- [ ] Create docker-compose.yml
- [ ] Set up production environment variables
- [ ] Configure gunicorn/uvicorn for production
- [ ] Set up CI/CD pipeline (optional)
Quick Reference
Free Models:
FREE_MODELS = {
    "chat": "meta-llama/llama-3.3-70b-instruct",
    "quiz": "deepseek/deepseek-r1-0528",
    "analysis": "stepfun/step-3.5-flash",
}
Commands:
# Development
uvicorn app.main:app --reload
# Migrations
alembic revision --autogenerate
alembic upgrade head
# Docker
docker build -t orbit-backend .
docker run -p 8000:8000 orbit-backend
---
Status Tracking
| Phase | Status | Notes |
|-------|--------|-------|
| 1. Project Setup | ⏳ Pending | |
| 2. Database Models | ⏳ Pending | |
| 3. Authentication | ⏳ Pending | |
| 4. PDF Management | ⏳ Pending | |
| 5. AI Integration | ⏳ Pending | |
| 6. Chat System | ⏳ Pending | |
| 7. PDF Extraction | ⏳ Pending | |
| 8. Analysis | ⏳ Pending | |
| 9. Quiz Generation | ⏳ Pending | |
| 10. Test/Results | ⏳ Pending | |
| 11. Testing | ⏳ Pending | |
| 12. Deployment | ⏳ Pending | |
---
