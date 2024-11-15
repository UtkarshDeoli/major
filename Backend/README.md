# Major Project

This is a Python project implementing a question-answering system for PDFs.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
cd src && python main.py
```

## Flow

1. Upload and process a PDF:
   ```
   POST /process-pdf
   ```
   This extracts text from the PDF and prepares it for question answering.

2. Ask a question about the PDF content:
   ```
   POST /ask-question
   ```
   This finds relevant context from the PDF and generates an answer using the LLM.

3. Stream a response to a question (recommended):
   ```
   POST /ask-question/stream
   ```
   This streams the LLM's response in real-time as it's being generated.

4. Check processing status:
   ```
   GET /status
   ```
   Verify if a PDF has been processed.

## Notes

- The system uses Sentence-BERT for finding relevant text in the PDF
- Questions are answered by a local LLM (llama3.2:1b) running at http://192.168.1.10:1234
- PDF processing must be done before asking questions
- The streaming endpoint returns JSON responses in a newline-delimited stream
