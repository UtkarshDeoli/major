# Orbit Multi-Document RAG System — Design Specification

**Date**: 2026-06-08
**Status**: Approved
**Scope**: Multi-document querying, hybrid search (BM25 + vector), source citations, multi-format document support

---

## 1. Overview & Goals

### Current State
Orbit's RAG system processes one PDF at a time using:
- `pypdf` for text extraction
- Regex-based paragraph splitting
- `sentence-transformers MiniLM-L6-v2` (384-dim) embeddings
- Simple cosine similarity against a single PDF's chunks
- JSON file storage per PDF

### Problems with Current System
1. **Single-document only** — Users must select one PDF per query. Can't ask "Compare my notes to the textbook."
2. **No source citations** — Answers don't cite which document or page the information came from.
3. **Only PDF support** — No `.txt`, `.md`, `.docx`, images, or scanned documents.
4. **Dumb chunking** — Regex split by paragraphs ignores document structure (headings, sections).
5. **Pure vector search** — Misses exact keyword matches critical for exam content (equation names, theorem names).
6. **No OCR** — Scanned PDFs and screenshots are completely unreadable.
7. **No document organization** — No tags, subjects, or collections to filter searches.

### Goals
1. **Multi-document querying** — Query across ALL uploaded documents by default. Optionally filter by doc, tag, or subject.
2. **Source citations** — Every answer cites exact document name, section, and page number.
3. **Multi-format support** — Handle `.pdf` (text + scanned), `.txt`, `.md`, `.docx`, images (PNG/JPG).
4. **Smart chunking** — Preserve heading hierarchy. Chunk by section, then paragraph, then sentence.
5. **Hybrid search** — Combine vector semantic search with BM25 keyword search via Reciprocal Rank Fusion (RRF).
6. **OCR for scanned content** — `pytesseract` for printed text, `easyocr` fallback for handwritten.
7. **Document metadata** — Tags, subjects, doc_type for filtering and organization.

---

## 2. Stack Decisions

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Embeddings** | `sentence-transformers` `all-MiniLM-L6-v2` (384-dim) | Fast on CPU, good quality, fits i3 server |
| **Vector Store** | **ChromaDB embedded mode** (`chromadb` Python package) | Native ANN search, no separate service, persists to disk |
| **Keyword Search** | `rank_bm25` (PyPI) | Pure Python, zero infra, fast enough for student corpora |
| **Fusion** | RRF with k=60 | Industry standard, no hyperparameter tuning needed |
| **Re-ranking** | Cross-encoder (v1.5, optional) | Better relevance ranking; deferred to Phase 2 |
| **OCR** | `pytesseract` + `easyocr` fallback | No API cost, runs on CPU, handles printed + handwritten |
| **LLM** | Gemini (existing) | No change needed |
| **Metadata DB** | MongoDB (existing) | Store chunk metadata, document info, user indexes |

### Key Decision: ChromaDB over JSON
ChromaDB embedded mode (`chromadb.Client()` with `Settings(anonymized_telemetry=False)`) stores vectors + metadata in a local SQLite-backed directory. Each user gets their own collection (`user_{hashed_user_id}`). This gives us:
- Native HNSW approximate nearest neighbor search
- Metadata filtering (`where={"doc_id": "..."}`)
- Incremental adds/deletes without rebuilding
- Zero network overhead (runs in the same Python process)

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DOCUMENT UPLOAD PIPELINE                          │
│                                                                      │
│  File (PDF/txt/md/docx/img)                                          │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────┐  Detect extension → map to doc_type               │
│  │ Detect Type │  pdf | txt | md | docx | image                      │
│  └──────┬──────┘                                                     │
│         ▼                                                            │
│  ┌─────────────┐  Extract raw text + page/heading info               │
│  │   Extract   │  PDF: pypdf/pdfplumber | docx: python-docx         │
│  │    Text     │  image: pytesseract → easyocr fallback              │
│  └──────┬──────┘                                                     │
│         ▼                                                            │
│  ┌─────────────┐  Split by headings → paragraphs → sentences          │
│  │   Chunk     │  Max 512 tokens per chunk, 50-token overlap        │
│  │             │  Each chunk gets: page, section, chunk_index       │
│  └──────┬──────┘                                                     │
│         ▼                                                            │
│  ┌─────────────┐  all-MiniLM-L6-v2, 384-dim embedding                │
│  │   Embed     │  One embedding per chunk                           │
│  └──────┬──────┘                                                     │
│         ▼                                                            │
│  ┌────────────────────────────────────────────┐                      │
│  │              DUAL STORAGE                   │                      │
│  │  ┌─────────────────┐  ┌─────────────────┐   │                      │
│  │  │   ChromaDB      │  │    MongoDB      │   │                      │
│  │  │  ─────────────   │  │  ─────────────   │   │                      │
│  │  │  Collection:      │  │  Collection:    │   │                      │
│  │  │  user_{hash}  │  │  document_chunks│   │                      │
│  │  │  ├── embedding   │  │  ├── user_id    │   │                      │
│  │  │  ├── content     │  │  ├── doc_id     │   │                      │
│  │  │  ├── metadata    │  │  ├── doc_name   │   │                      │
│  │  │  │   (doc_id,   │  │  ├── chunk_index│   │                      │
│  │  │  │    page,     │  │  ├── page       │   │                      │
│  │  │  │    section)  │  │  ├── section    │   │                      │
│  │  │  └── id          │  │  ├── doc_type   │   │                      │
│  │  │                 │  │  ├── subject    │   │                      │
│  │  │                 │  │  ├── tags[]      │   │                      │
│  │  │                 │  │  └── chroma_id   │   │                      │
│  │  └─────────────────┘  └─────────────────┘   │                      │
│  └────────────────────────────────────────────┘                      │
│                                                                      │
│  ┌─────────────┐  Add chunks to user's BM25 index in-memory          │
│  │ Update BM25 │  Rebuild if index doesn't exist yet                  │
│  │    Index    │                                                                     │
│  └─────────────┘                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE                                  │
│                                                                      │
│  User Question ──→ User ID + Optional Filters (doc_ids, tags, subject)│
│                                                                      │
│       │                                                              │
│       ▼                                                              │
│  ┌────────────┐     ┌────────────┐                                 │
│  │   Vector   │     │   BM25     │                                 │
│  │   Search   │     │   Search   │                                 │
│  │  ChromaDB  │     │  In-Memory │                                 │
│  │  top_k=20  │     │  top_k=20  │                                 │
│  └─────┬──────┘     └─────┬──────┘                                 │
│        │                  │                                          │
│        └────────┬─────────┘                                          │
│                 ▼                                                    │
│       ┌──────────────────┐                                           │
│       │   RRF Fusion     │  score = Σ(1 / (60 + rank))              │
│       │    (k = 60)      │                                           │
│       └────────┬─────────┘                                           │
│                ▼                                                     │
│       ┌──────────────────┐                                           │
│       │  Fetch Metadata   │  From MongoDB by chunk IDs               │
│       │  (pages, sections)│                                          │
│       └────────┬─────────┘                                           │
│                ▼                                                     │
│       ┌──────────────────┐                                           │
│       │ Build Context   │  With inline citations:                   │
│       │ with Citations  │  "[1] Physics.pdf, Thermodynamics, P.12"  │
│       └────────┬─────────┘                                           │
│                ▼                                                     │
│            Gemini LLM                                                │
│                │                                                     │
│                ▼                                                     │
│         Answer + Source List                                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Models

### 4.1 MongoDB Collection: `document_chunks`

```javascript
{
  _id: ObjectId,
  
  // Ownership
  user_id: String,              // User identifier string (from existing auth system, typically email)
  
  // Document linkage
  doc_id: String,               // ObjectId of document in `pdfs` collection
  doc_name: String,             // Human-readable: "Physics_Notes.pdf"
  
  // Chunk ordering (for context expansion)
  chunk_index: Number,          // 0, 1, 2... within this document
  
  // ChromaDB linkage
  chroma_id: String,            // UUID generated by ChromaDB.add()
  
  // Content (stored in ChromaDB too, but mirrored here for BM25)
  content: String,              // The actual text chunk (max ~2000 chars)
  
  // Source metadata (for citations — format-specific)
  page: Number,                 // Page number (PDFs only; null for txt/md/docx/images)
  section: String,              // Heading/section name (universal)
  locator: String,              // Format-specific position: "line 42" | "paragraph 5" | "page 3"
  total_pages: Number,          // Total pages in source document (null for non-paginated)
  
  // Document metadata (for filtering)
  doc_type: String,             // "pdf" | "txt" | "md" | "docx" | "image" | "scanned_pdf"
  subject: String,              // User-defined: "Physics", "Mathematics"
  tags: [String],               // ["semester-3", "exam-prep", "important"]
  
  // OCR metadata (if applicable)
  ocr_confidence: Number,       // 0.0-1.0 (null for non-OCR docs)
  ocr_engine: String,           // "tesseract" | "easyocr" | null
  
  // Indexing
  created_at: Date
}
```

**Indexes to create:**
```javascript
db.document_chunks.createIndex({ user_id: 1, doc_id: 1 })
db.document_chunks.createIndex({ user_id: 1, subject: 1 })
db.document_chunks.createIndex({ user_id: 1, tags: 1 })
db.document_chunks.createIndex({ chroma_id: 1 }, { unique: true })
```

### 4.2 MongoDB Collection: `pdfs` (Extended)

Add these fields to the existing `pdfs` collection:

```javascript
{
  // ... existing fields (filename, size, user_id, file_path, etc.) ...
  
  // Processing status
  processed: Boolean,           // true when chunking + embedding complete
  processing_error: String,     // null or error message
  chunk_count: Number,          // Total chunks extracted
  
  // Document metadata
  doc_type: String,             // "pdf" | "txt" | "md" | "docx" | "image"
  subject: String,              // User-defined subject
  tags: [String],               // User-defined tags
  
  // Source info
  page_count: Number,           // Total pages (if applicable)
  has_scanned_pages: Boolean,   // true if OCR was used
  
  // Storage
  vector_db_path: String,       // DEPRECATED: remove in migration
  chroma_collection: String      // "user_{hash}"
}
```

### 4.3 ChromaDB Collection Schema

Each user gets one collection: `user_{hashed_user_id}`

```python
collection = chroma_client.get_or_create_collection(
    name=f"user_{{get_collection_suffix(user_id)}}",
    metadata={"user_id": user_id},
    embedding_function=embedding_function  # all-MiniLM-L6-v2
)

# Add chunks:
collection.add(
    ids=[chunk.chroma_id for chunk in chunks],
    embeddings=[chunk.embedding for chunk in chunks],
    documents=[chunk.content for chunk in chunks],
    metadatas=[{
        "doc_id": chunk.doc_id,
        "doc_name": chunk.doc_name,
        "page": chunk.page,
        "section": chunk.section,
        "chunk_index": chunk.chunk_index,
        "subject": chunk.subject,
        "tags": chunk.tags
    } for chunk in chunks]
)

# Query:
collection.query(
    query_embeddings=[question_embedding],
    n_results=20,
    where={"doc_id": {"$in": filtered_doc_ids}}  # optional filter
)
```

---

## 5. Document Processing Pipeline

### 5.1 Document Type Detection

```python
DOC_TYPE_MAP = {
    '.pdf': 'pdf',
    '.txt': 'txt',
    '.md': 'md',
    '.docx': 'docx',
    '.png': 'image', '.jpg': 'image', '.jpeg': 'image', 
    '.webp': 'image', '.bmp': 'image'
}

def detect_doc_type(filename: str) -> str:
    ext = os.path.splitext(filename.lower())[1]
    return DOC_TYPE_MAP.get(ext, 'unknown')
```

### 5.2 Text Extraction by Type

| doc_type | Primary Extractor | Fallback | Notes |
|----------|------------------|----------|-------|
| `pdf` | `pdfplumber` | `pypdf` | Extract text + page numbers + bounding boxes |
| `scanned_pdf` | `pytesseract` | `easyocr` | Render PDF pages to images, then OCR |
| `txt` | Native `open()` | — | Read as UTF-8 |
| `md` | Native `open()` | — | Preserve heading markers for chunking |
| `docx` | `python-docx` | — | Extract paragraphs, detect heading styles |
| `image` | `pytesseract` | `easyocr` | Direct OCR on image |

### 5.3 Scanned PDF Detection

Before extracting a PDF, check if it contains text or is image-based:

```python
def is_scanned_pdf(file_path: str, text_threshold: int = 50) -> Tuple[bool, float]:
    """
    Check if PDF is scanned (image-based) using stratified sampling.
    Samples pages from start, 25%, 50%, 75%, and end for robust detection.
    Returns: (is_scanned: bool, confidence: float)
    """
    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)
        if total_pages == 0:
            return True, 1.0  # Empty PDF, assume scanned
        
        # Determine sample indices based on document length
        if total_pages <= 3:
            sample_indices = list(range(total_pages))
        elif total_pages <= 10:
            sample_indices = [0, 1, 2, total_pages - 1]  # first 3 + last
        else:
            sample_indices = [
                0, 1, 2,                    # Start
                total_pages // 4,         # 25%
                total_pages // 2,         # 50%
                (3 * total_pages) // 4,   # 75%
                total_pages - 2,          # Near end
                total_pages - 1           # Last page
            ]
        
        sample_indices = sorted(set(i for i in sample_indices if 0 <= i < total_pages))
        
        text_pages = 0
        for i in sample_indices:
            page = pdf.pages[i]
            text = page.extract_text() or ""
            if len(text.strip()) > text_threshold:
                text_pages += 1
        
        text_ratio = text_pages / len(sample_indices)
        is_scanned = text_ratio < 0.3  # Less than 30% text pages = scanned
        confidence = 1.0 - text_ratio
        
        return is_scanned, confidence
```

If `is_scanned_pdf()` returns `is_scanned=True` (confidence > 0.7), route to OCR pipeline. If confidence is low (0.3–0.7), run BOTH text extraction and OCR, then merge results.

### 5.4 OCR Pipeline

```python
async def ocr_pdf_pages(file_path: str) -> List[OCRPage]:
    """Convert PDF pages to images, then OCR each page."""
    from pdf2image import convert_from_path
    
    images = convert_from_path(file_path, dpi=200)
    pages = []
    
    for i, image in enumerate(images):
        # Try pytesseract first with confidence data
        tesseract_data = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT
        )
        
        # Extract text and calculate mean word confidence
        words = [word for word, conf in zip(tesseract_data["text"], tesseract_data["conf"]) 
                 if int(conf) > 0 and word.strip()]
        confidences = [int(c) for c in tesseract_data["conf"] if int(c) > 0]
        
        text = " ".join(words)
        confidence = np.mean(confidences) / 100.0 if confidences else 0.0
        engine = "tesseract"
        
        # Fallback to easyocr if mean confidence is low (< 50%) or text is too short
        if confidence < 0.5 or len(text.strip()) < 20:
            easyocr_result = easyocr_reader.readtext(np.array(image))
            if easyocr_result:
                text = " ".join([r[1] for r in easyocr_result])
                confidence = np.mean([r[2] for r in easyocr_result])
                engine = "easyocr"
        
        pages.append(OCRPage(
            page_num=i + 1,
            text=text,
            confidence=confidence,
            engine=engine
        ))
    
    return pages
```

### 5.5 Hierarchical Chunking

```python
@dataclass
class Chunk:
    content: str
    page: int
    section: str
    chunk_index: int
    char_start: int
    char_end: int

MAX_CHUNK_TOKENS = 512
OVERLAP_TOKENS = 50

def chunk_document(text: str, headings: List[Heading], 
                   doc_type: str, total_pages: int) -> List[Chunk]:
    """
    1. Split document by headings into sections
    2. Within each section, split into paragraphs
    3. If paragraph > MAX_CHUNK_TOKENS, split by sentences
    4. Add overlap between consecutive chunks
    """
    chunks = []
    chunk_index = 0
    
    # Step 1: Split by headings
    sections = split_by_headings(text, headings)
    
    for section in sections:
        # Step 2: Split section into paragraphs
        paragraphs = split_paragraphs(section.content)
        
        for para in paragraphs:
            para_tokens = estimate_tokens(para)
            
            if para_tokens <= MAX_CHUNK_TOKENS:
                # Paragraph fits in one chunk
                chunks.append(Chunk(
                    content=para,
                    page=estimate_page(para, section.start_pos, total_pages),
                    section=section.heading,
                    chunk_index=chunk_index,
                    char_start=section.start_pos,
                    char_end=section.start_pos + len(para)
                ))
                chunk_index += 1
            else:
                # Step 3: Split by sentences
                sentences = split_sentences(para)
                current_chunk = []
                current_tokens = 0
                
                for sentence in sentences:
                    sent_tokens = estimate_tokens(sentence)
                    
                    if current_tokens + sent_tokens > MAX_CHUNK_TOKENS:
                        # Flush current chunk
                        chunks.append(build_chunk(current_chunk, section, chunk_index))
                        chunk_index += 1
                        
                        # Step 4: Start new chunk with overlap
                        overlap = get_overlap_tokens(current_chunk, OVERLAP_TOKENS)
                        current_chunk = overlap + [sentence]
                        current_tokens = estimate_tokens(" ".join(current_chunk))
                    else:
                        current_chunk.append(sentence)
                        current_tokens += sent_tokens
                
                # Flush remaining
                if current_chunk:
                    chunks.append(build_chunk(current_chunk, section, chunk_index))
                    chunk_index += 1
    
    return chunks
```

### 5.6 Heading Detection

```python
def detect_headings(text: str, doc_type: str) -> List[Heading]:
    """Detect headings based on document type."""
    
    if doc_type in ['md', 'txt']:
        # Markdown-style: # ## ###
        pattern = r'^(#{1,4}\s+.+)$'
        return extract_by_pattern(text, pattern)
    
    elif doc_type == 'pdf':
        # Heuristic: short lines, all caps, or bold formatting
        lines = text.split('\n')
        headings = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if is_heading_heuristic(stripped, lines, i):
                headings.append(Heading(text=stripped, line_index=i))
        return headings
    
    elif doc_type == 'docx':
        # python-docx provides heading styles
        return extract_docx_headings(text)
    
    return []

def is_heading_heuristic(line: str, context: List[str], index: int) -> bool:
    """Heuristic to detect headings in plain text."""
    if not line or len(line) > 100:
        return False
    
    # All caps
    if line.upper() == line and len(line.split()) <= 10:
        return True
    
    # Numbered section: "4.2 Thermodynamics" or "Chapter 3"
    if re.match(r'^(\d+(\.\d+)*\s+[A-Z]|Chapter\s+\d+|Section\s+\d+)', line):
        return True
    
    # Short line followed by longer paragraph
    if index + 1 < len(context) and len(line) < len(context[index + 1].strip()) * 0.5:
        return True
    
    return False
```

---

## 6. Query Pipeline

### 6.1 Vector Search (ChromaDB)

```python
async def vector_search(
    user_id: str, 
    question: str, 
    doc_ids: Optional[List[str]] = None,
    subject: Optional[str] = None,
    tags: Optional[List[str]] = None,
    top_k: int = 20
) -> List[SearchResult]:
    """Search user's ChromaDB collection for semantically similar chunks."""
    
    collection = get_user_collection(user_id)
    embedding = embedding_model.encode(question).tolist()
    
    # Build where clause for optional filtering
    # Note: user_id is NOT stored in chunk metadata; tenant isolation is enforced
    # by having separate ChromaDB collections per user. Only chunk-level filters here.
    where_clause = {}
    if doc_ids:
        where_clause["doc_id"] = {"$in": doc_ids}
    if subject:
        where_clause["subject"] = subject
    if tags:
        where_clause["tags"] = {"$in": tags}
    
    # If no filters, pass None (ChromaDB convention)
    where_filter = where_clause if where_clause else None
    
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        where=where_filter,
        include=["metadatas", "documents", "distances"]
    )
    
    return [
        SearchResult(
            chroma_id=results["ids"][0][i],
            score=1 - results["distances"][0][i],  # Convert distance to similarity
            content=results["documents"][0][i],
            metadata=results["metadatas"][0][i]
        )
        for i in range(len(results["ids"][0]))
    ]
```

### 6.2 BM25 Search

```python
class UserBM25Index:
    """Per-user BM25 index, rebuilt on startup, incrementally updated."""
    
    def __init__(self):
        self.indexes: Dict[str, Optional[BM25Okapi]] = {}  # user_id -> index
        self.chunk_maps: Dict[str, Dict[int, str]] = {}  # user_id -> {bm25_index -> chroma_id}
        self.chunk_metadata: Dict[str, Dict[str, Dict]] = {}  # user_id -> {chroma_id -> {doc_id, subject, tags}}
    
    async def build_index(self, user_id: str):
        """Build BM25 index for a user from MongoDB chunks."""
        chunks = await get_user_chunks_for_bm25(user_id)
        
        if not chunks:
            self.indexes[user_id] = None
            self.chunk_maps[user_id] = {}
            self.chunk_metadata[user_id] = {}
            return
        
        tokenized_corpus = [self.tokenize(c.content) for c in chunks]
        self.indexes[user_id] = BM25Okapi(tokenized_corpus)
        self.chunk_maps[user_id] = {i: c.chroma_id for i, c in enumerate(chunks)}
        
        # Build metadata map for post-search filtering
        self.chunk_metadata[user_id] = {
            c.chroma_id: {
                "doc_id": c.doc_id,
                "subject": c.subject,
                "tags": c.tags
            }
            for c in chunks
        }
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenizer: lowercase, strip punctuation, split on whitespace."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def search(self, user_id: str, query: str, top_k: int = 20,
               doc_ids: Optional[List[str]] = None,
               subject: Optional[str] = None,
               tags: Optional[List[str]] = None) -> List[str]:
        """Return list of chroma_ids, respecting filters."""
        if user_id not in self.indexes or self.indexes[user_id] is None:
            return []
        
        tokenized_query = self.tokenize(query)
        scores = self.indexes[user_id].get_scores(tokenized_query)
        
        # Apply filters post-search by zeroing out excluded chunks
        for i, chroma_id in enumerate(self.chunk_maps[user_id].values()):
            chunk = self.chunk_metadata[user_id].get(chroma_id)
            if not chunk:
                scores[i] = 0
                continue
            if doc_ids and chunk["doc_id"] not in doc_ids:
                scores[i] = 0
            if subject and chunk.get("subject") != subject:
                scores[i] = 0
            if tags and not any(t in chunk.get("tags", []) for t in tags):
                scores[i] = 0
        
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [self.chunk_maps[user_id][i] for i in top_indices if scores[i] > 0]
    
    def add_document(self, user_id: str, chunks: List[DocumentChunk]):
        """Incrementally add chunks to existing index."""
        # For simplicity, rebuild the index
        # With rank_bm25, incremental updates are complex; rebuild is fine for <10k chunks
        asyncio.create_task(self.build_index(user_id))
```

### 6.3 Reciprocal Rank Fusion (RRF)

```python
def reciprocal_rank_fusion(
    vector_results: List[SearchResult],
    bm25_results: List[str],
    k: int = 60
) -> Dict[str, float]:
    """
    Fuse two ranked lists using Reciprocal Rank Fusion.
    
    score(d) = Σ(1 / (k + rank_d))
    
    Where rank_d is the 0-based rank of document d in each list.
    Documents not in a list get score 0 from that list.
    """
    scores = {}
    
    # Add vector scores
    for rank, result in enumerate(vector_results):
        doc_id = result.chroma_id
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    
    # Add BM25 scores
    for rank, doc_id in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    
    return scores
```

### 6.4 Full Query Orchestrator

```python
async def query_documents(
    user_id: str,
    question: str,
    doc_ids: Optional[List[str]] = None,
    subject: Optional[str] = None,
    tags: Optional[List[str]] = None,
    top_k: int = 5
) -> Tuple[str, List[Source], List[DocumentChunk]]:
    """
    Full query pipeline:
    1. Vector search (ChromaDB)
    2. BM25 search
    3. RRF fusion
    4. Fetch metadata from MongoDB
    5. Build context with citations
    
    Returns: (context_string, sources_list, chunks)
    """
    
    # Step 1: Vector search (with all filters)
    vector_results = await vector_search(
        user_id, question, doc_ids=doc_ids, subject=subject, tags=tags, top_k=20
    )
    
    # Step 2: BM25 search (with all filters)
    bm25_results = bm25_index.search(
        user_id, question, top_k=20,
        doc_ids=doc_ids, subject=subject, tags=tags
    )
    
    # Step 3: RRF fusion
    fused_scores = reciprocal_rank_fusion(vector_results, bm25_results, k=60)
    
    # Sort by fused score, take top_k
    top_chroma_ids = sorted(
        fused_scores.keys(),
        key=lambda x: fused_scores[x],
        reverse=True
    )[:top_k]
    
    # Step 4: Fetch full metadata from MongoDB
    chunks = await get_chunks_by_chroma_ids(top_chroma_ids)
    
    # Optional: Expand context with adjacent chunks
    expanded_chunks = await expand_with_adjacent(chunks, window=1)
    
    # Step 5: Build context string
    context_parts = []
    sources = []
    
    for i, chunk in enumerate(expanded_chunks, 1):
        citation = f"[{i}] {chunk.doc_name}"
        if chunk.section:
            citation += f", {chunk.section}"
        # Format-specific position info
        if chunk.page:
            citation += f", Page {chunk.page}"
        elif chunk.locator:
            citation += f", {chunk.locator}"
        
        context_parts.append(f"{citation}\n{chunk.content}")
        sources.append(Source(
            index=i,
            doc_name=chunk.doc_name,
            page=chunk.page,
            section=chunk.section,
            locator=chunk.locator,
            chroma_id=chunk.chroma_id
        ))
    
    context = "\n\n---\n\n".join(context_parts)
    
    return context, sources, expanded_chunks
```

### 6.5 Context Expansion (Optional)

```python
async def expand_with_adjacent(
    chunks: List[DocumentChunk], 
    window: int = 1
) -> List[DocumentChunk]:
    """Include adjacent chunks for better context."""
    expanded = []
    
    for chunk in chunks:
        expanded.append(chunk)
        
        # Fetch previous and next chunks
        adjacent = await get_adjacent_chunks(
            chunk.doc_id, 
            chunk.chunk_index, 
            window=window
        )
        
        for adj in adjacent:
            if adj not in expanded:
                expanded.append(adj)
    
    return expanded
```

---

## 7. LLM Prompt with Citations

```python
def build_rag_prompt(
    question: str, 
    context: str, 
    sources: List[Source]
) -> str:
    """Build the prompt with inline citation requirements."""
    
    sources_list = "\n".join([
        f"[{s.index}] {s.doc_name}" + 
        (f", {s.section}" if s.section else "") + 
        (f", Page {s.page}" if s.page else "")
        for s in sources
    ])
    
    return f"""You are an AI tutor helping a student prepare for exams. Answer the following question STRICTLY based on the provided document excerpts.

Rules:
- Start directly with the answer. No introductions like "Based on the documents..."
- Cite sources inline using [index] format. Example: "The photoelectric effect [1] demonstrates that light behaves as particles."
- Focus on key points important for exams. Break down complex ideas simply.
- If the context doesn't contain enough information, say: "I don't have enough information about that in your uploaded documents."
- Do not make up information not present in the excerpts.

Sources:
{sources_list}

Document Excerpts:
{context}

Question: {question}

Answer (with inline citations):"""
```

### 7.1 Response Parsing

```python
def parse_cited_response(response: str) -> Tuple[str, List[int]]:
    """
    Extract which source indices were actually cited in the answer.
    This lets the frontend show only relevant sources.
    """
    cited_indices = list(set(map(int, re.findall(r'\[(\d+)\]', response))))
    return response, cited_indices
```

---

## 8. API Changes

### 8.1 New Endpoint: Upload Any Document

```python
@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    subject: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),  # comma-separated
    user_id: str = Depends(get_current_user)
) -> DocumentUploadResponse:
    """
    Upload any document type (pdf, txt, md, docx, image).
    
    - Detects doc_type from extension
    - Extracts text (with OCR if needed)
    - Hierarchical chunking
    - Embeds and stores in ChromaDB + MongoDB
    - Updates BM25 index
    
    Returns: DocumentUploadResponse with doc_id, chunk_count, doc_type
    """
```

**Response:**
```json
{
  "id": "68f123abc...",
  "filename": "Physics_Notes.pdf",
  "doc_type": "pdf",
  "size": 245000,
  "chunk_count": 42,
  "page_count": 15,
  "has_scanned_pages": false,
  "subject": "Physics",
  "tags": ["semester-3", "exam-prep"],
  "processed": true,
  "upload_date": "2026-06-08T14:30:00Z"
}
```

### 8.2 Updated Endpoint: Ask Question (Multi-Document)

```python
@router.post("/questions/ask")
async def ask_question(
    request: QuestionRequest,
    user_id: str = Depends(get_current_user)
) -> QuestionResponse:
    """
    Ask a question across all documents (or filtered subset).
    
    If doc_ids is empty/null, searches ALL user documents.
    """

class QuestionRequest(BaseModel):
    question: str
    doc_ids: Optional[List[str]] = None      # null = search all docs
    subject: Optional[str] = None             # filter by subject
    tags: Optional[List[str]] = None           # filter by tags
    stream: bool = False
    top_k: int = 5                           # chunks to include in context

class QuestionResponse(BaseModel):
    answer: str
    sources: List[Source]
    context: str                              # For transparency/debugging
```

**Example Request:**
```json
{
  "question": "Explain the photoelectric effect and its significance in quantum mechanics.",
  "doc_ids": null,
  "subject": "Physics",
  "tags": ["important"],
  "top_k": 5
}
```

**Example Response:**
```json
{
  "answer": "The photoelectric effect [1] is the emission of electrons when light hits a material. Einstein explained it in 1905 by proposing light consists of photons [2], each with energy E = hν. This was crucial evidence for quantum theory [3] because it showed light has particle-like properties.",
  "sources": [
    {"index": 1, "doc_name": "Physics_Textbook.pdf", "section": "7.1 The Photoelectric Effect", "page": 234},
    {"index": 2, "doc_name": "Physics_Textbook.pdf", "section": "7.2 Einstein's Explanation", "page": 238},
    {"index": 3, "doc_name": "Class_Notes_Lecture_5.pdf", "section": "Quantum Mechanics Introduction", "page": 12}
  ],
  "context": "[1] Physics_Textbook.pdf, 7.1 The Photoelectric Effect, Page 234\nWhen light of sufficient frequency shines on a metal surface...\n\n---\n\n[2] Physics_Textbook.pdf, 7.2 Einstein's Explanation, Page 238\nEinstein proposed that light energy is quantized..."
}
```

### 8.3 New Endpoint: List User Documents

```python
@router.get("/documents")
async def list_documents(
    subject: Optional[str] = None,
    tags: Optional[List[str]] = None,
    user_id: str = Depends(get_current_user)
) -> DocumentListResponse:
    """List all documents with filtering by subject/tags."""
```

### 8.4 New Endpoint: Document Collections

```python
@router.post("/documents/{doc_id}/tags")
async def update_document_tags(
    doc_id: str,
    tags: List[str],
    user_id: str = Depends(get_current_user)
):
    """Update tags for a document. Propagates to all chunks."""
```

### 8.5 Backward Compatibility & Migration

The current API uses `pdf_id` (singular string) for single-document queries. The new multi-document API uses `doc_ids` (list of strings).

**Phase 1 (Deploy): Support both parameters**

```python
class QuestionRequest(BaseModel):
    question: str
    pdf_id: Optional[str] = None        # DEPRECATED: old single-document field
    doc_ids: Optional[List[str]] = None # NEW: multi-document field
    subject: Optional[str] = None
    tags: Optional[List[str]] = None
    stream: bool = False
    top_k: int = 5

async def ask_question(request: QuestionRequest, user_id: str = Depends(get_current_user)):
    # Normalize deprecated pdf_id into doc_ids
    if request.pdf_id and not request.doc_ids:
        request.doc_ids = [request.pdf_id]
    
    # If neither provided, search ALL user documents (new default)
    doc_ids = request.doc_ids  # may be None
    
    # Proceed with query_documents()
    context, sources, chunks = await query_documents(
        user_id=user_id,
        question=request.question,
        doc_ids=doc_ids,
        subject=request.subject,
        tags=request.tags,
        top_k=request.top_k
    )
    
    # ... rest of handler
```

**Frontend Migration:**
1. **Week 1**: Frontend can continue sending `pdf_id`. Backend handles it.
2. **Week 2**: Frontend updates to send `doc_ids` (array) when user selects documents.
3. **Week 3**: Frontend adds "Search All Documents" toggle (sends `doc_ids: null`).
4. **Week 4**: Remove `pdf_id` from frontend. After 1 month, remove backend support.

**Existing Chat Sessions:**
- Sessions created before migration store `pdf_id` in MongoDB.
- On session resume, if `pdf_id` exists but no `doc_ids`, treat as `doc_ids=[pdf_id]`.
- Update `chat_sessions` schema to store `doc_ids: List[str]` instead of `pdf_id: String`.

**Data Migration:**
- No breaking data migration needed.
- Existing `pdfs` documents will be backfilled on first query (see Section 11).
- Old `vector_db_path` JSON files remain untouched; ignored by new code.

---

## 9. Implementation Phases

### Phase 1: Core Multi-Document RAG (Week 1)
**Goal**: Ship multi-document querying with hybrid search.

- [ ] Add `chromadb` to requirements.txt
- [ ] Create `document_chunks` MongoDB collection + indexes
- [ ] Build `DocumentProcessor` service:
  - [ ] Type detection
  - [ ] PDF text extraction (existing `pdfplumber` path)
  - [ ] Hierarchical chunking
  - [ ] Embedding generation
- [ ] Build `VectorStore` service (ChromaDB wrapper):
  - [ ] User collections (`user_{hashed_user_id}`)
  - [ ] Add chunks with metadata
  - [ ] Query with optional filters
- [ ] Build `BM25Index` service:
  - [ ] Per-user index
  - [ ] Rebuild on startup
  - [ ] Incremental update on upload
- [ ] Build `QueryEngine` service:
  - [ ] Vector search
  - [ ] BM25 search
  - [ ] RRF fusion
  - [ ] Context building with citations
- [ ] Update `llm_service.py`:
  - [ ] New prompt template with citations
  - [ ] Source tracking in response
- [ ] Update `pdf_router.py` → rename to `document_router.py`:
  - [ ] New `/documents/upload` endpoint
  - [ ] Update `/questions/ask` to support multi-doc
- [ ] Update `data_store.py`:
  - [ ] Chunk CRUD operations
  - [ ] Tag/subject filtering

### Phase 2: Multi-Format + OCR (Week 2)
**Goal**: Support all document types and scanned content.

- [ ] Add `pytesseract`, `easyocr`, `pdf2image`, `python-docx` to requirements
- [ ] Implement OCR pipeline:
  - [ ] Scanned PDF detection
  - [ ] Page-to-image conversion
  - [ ] Tesseract OCR
  - [ ] EasyOCR fallback
- [ ] Implement multi-format extractors:
  - [ ] `.txt` / `.md` reader
  - [ ] `.docx` extractor
  - [ ] Image OCR direct upload
- [ ] Update chunking for each format:
  - [ ] Markdown heading detection
  - [ ] DOCX style-based headings
- [ ] Add OCR confidence scores to chunk metadata
- [ ] Frontend: Support image upload in UI

### Phase 3: UX Enhancements (Week 3)
**Goal**: NotebookLM-level user experience.

- [ ] Document collections (group by subject)
- [ ] Tag management UI
- [ ] Source citation cards in chat UI
- [ ] Click-to-jump-to-source in PDF viewer
- [ ] Cross-encoder re-ranking (optional, if needed)
- [ ] Document similarity visualization
- [ ] "Study session" mode: pre-load specific collections

---

## 10. Testing Strategy

### Unit Tests
- [ ] Document type detection
- [ ] Heading detection (each format)
- [ ] Chunking logic (edge cases: short docs, no headings)
- [ ] RRF fusion math
- [ ] BM25 tokenizer
- [ ] Prompt building

### Integration Tests
- [ ] Full upload → query flow
- [ ] Multi-document query returns sources from different docs
- [ ] BM25 + vector both contribute to results
- [ ] Filtering by doc_id, subject, tags
- [ ] OCR pipeline for scanned PDF

### Load Tests
- [ ] Query latency with 10, 50, 100 documents
- [ ] Memory usage with large BM25 indexes
- [ ] ChromaDB query performance

---

## 11. Migration Plan

### From Existing PDF System
1. **Keep existing `pdfs` collection** — Add new fields (`doc_type`, `chunk_count`, `subject`, `tags`)
2. **Create `document_chunks` collection** — New collection, no migration needed
3. **Deprecate `vector_db_path` in `pdfs`** — Existing JSON files ignored; re-process on next upload
4. **Backfill existing PDFs** — On first query after deploy, if no chunks exist for a PDF, process it in the background

### Zero-Downtime Deploy
1. Deploy new code alongside existing
2. New uploads go through new pipeline
3. Existing PDFs work via fallback (old JSON path) until backfilled
4. After 1 week, remove fallback code

---

## 12. Performance Budget

| Metric | Target | Notes |
|--------|--------|-------|
| Upload processing | <30s for 50-page PDF | Async background task |
| Query latency | <2s end-to-end | Vector search + BM25 + LLM call |
| Memory per user | <100MB | BM25 index + ChromaDB cache |
| Storage per doc | ~2x original size | ChromaDB + MongoDB metadata |
| Concurrent users | 50+ | FastAPI async + ChromaDB embedded |

---

## 13. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| ChromaDB embedded mode doesn't scale | Low | High | Can migrate to ChromaDB server mode or Qdrant later |
| OCR accuracy poor on handwritten notes | Medium | Medium | EasyOCR fallback; warn users about low confidence |
| BM25 index rebuild slow at scale | Low | Medium | Add pickle serialization for large indexes |
| MongoDB query slow with many chunks | Low | Medium | Ensure indexes on `user_id + doc_id` |
| Chunking loses math/equations | Medium | High | Preserve LaTeX/math in chunks; don't strip symbols |

---

## 14. Appendix

### A. Dependencies to Add

```
chromadb>=0.4.0
rank_bm25>=0.2.0
pytesseract>=0.3.10
pdf2image>=1.16.3
pillow>=10.0.0
easyocr>=1.7.0
python-docx>=0.8.11
```

### B. Environment Variables

```bash
# Existing
MONGODB_URL=...
GEMINI_API_KEY=...
SECRET_KEY=...

# New
CHROMA_DB_PATH=./chroma_db       # ChromaDB persistence directory
TESSERACT_CMD=/usr/bin/tesseract  # Tesseract binary path (optional)
```

### C. ChromaDB Collection Naming

```python
import hashlib

# Naming convention: user_{hashed_user_id}
# Uses a stable SHA-256 hash of the user_id to guarantee uniqueness
# and avoid collisions from email sanitization (e.g. a@b.com vs a_b@com).

def get_collection_name(user_id: str) -> str:
    hashed = hashlib.sha256(user_id.encode()).hexdigest()[:32]
    return f"user_{hashed}"
```

---

**End of Design Specification**
