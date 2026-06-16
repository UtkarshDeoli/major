# Major Project

This is a Python project implementing a question-answering system for PDFs with AI-powered question paper analysis.

## Features

- **PDF Upload & Processing**: Upload and process PDF documents for Q&A
- **Question Answering**: Ask questions about uploaded PDF content
- **Question Paper Analysis**: AI-powered analysis of syllabus and previous year question papers using Gemini 2.0 Flash
- **Chat Sessions**: Manage conversation history
- **User Authentication**: Secure user management with JWT

## Installation

1. Clone the repository and navigate to the Backend directory
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env and add your Gemini API key
   ```

## Environment Setup

Create a `.env` file with the following variables:

```env
# MongoDB Configuration
MONGODB_URL=your_mongodb_connection_string
MONGODB_DB_NAME=your_database_name

# JWT Configuration
SECRET_KEY=your_secret_key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Gemini API Configuration
# Get your API key from: https://ai.google.dev/tutorials/setup
GEMINI_API_KEY=your_gemini_api_key_here
```

## Usage

```bash  
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8001
```

## API Endpoints

### Authentication
- `POST /auth/signup` - Register a new user
- `POST /auth/login` - Login user

### PDF Management
- `POST /pdfs/upload` - Upload and process a PDF
- `GET /pdfs/` - List user's PDFs
- `GET /pdfs/{pdf_id}` - Get PDF metadata
- `GET /pdfs/{pdf_id}/download` - Download PDF file

### Question Answering
- `POST /questions/ask` - Ask a question about PDF content
- `POST /questions/ask/stream` - Stream response to a question
- `POST /questions/sessions` - Create a new chat session
- `GET /questions/sessions` - List user's chat sessions
- `POST /questions/sessions/{session_id}/messages` - Add message to session

### Question Paper Analysis (NEW)
- `POST /analysis/question-papers` - Analyze question papers using Gemini AI

## Question Paper Analysis

The new question paper analysis feature uses Google's Gemini 2.0 Flash model to:

1. **Analyze syllabus and previous year question papers**
2. **Identify question patterns and weightage distribution**
3. **Generate unit-wise analysis with difficulty levels**
4. **Provide focus areas and preparation strategies**
5. **Create sample questions based on identified patterns**

### Usage Example:

```json
{
  "syllabus_pdf_id": "your_syllabus_pdf_id",
  "question_paper_pdf_ids": ["paper1_id", "paper2_id", "paper3_id"]
}
```

## Notes

- The system uses Sentence-BERT for finding relevant text in PDFs
- Question paper analysis requires a valid Gemini API key
- PDF processing must be completed before analysis
- All endpoints require authentication except signup and login
