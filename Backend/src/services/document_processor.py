import os
import re
import io
from typing import List, Dict, Any, Optional, Tuple
from pypdf import PdfReader

DOC_TYPE_MAP = {
    '.pdf': 'pdf',
    '.txt': 'txt',
    '.md': 'md',
    '.docx': 'docx',
    '.png': 'image', '.jpg': 'image', '.jpeg': 'image',
    '.webp': 'image', '.bmp': 'image'
}


def detect_doc_type(filename: str) -> str:
    """Detect document type from file extension."""
    ext = os.path.splitext(filename.lower())[1]
    return DOC_TYPE_MAP.get(ext, 'unknown')


def extract_text_from_pdf(file_content: bytes) -> Tuple[str, int]:
    """Extract text from a text-based PDF."""
    pdf_reader = PdfReader(io.BytesIO(file_content))
    page_count = len(pdf_reader.pages)

    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n\n"

    return text, page_count


def extract_text_from_txt(file_content: bytes) -> str:
    """Extract text from a .txt file."""
    return file_content.decode('utf-8', errors='ignore')


def extract_text_from_md(file_content: bytes) -> str:
    """Extract text from a .md file."""
    return file_content.decode('utf-8', errors='ignore')


def detect_headings(text: str, doc_type: str) -> List[Dict]:
    """Detect headings in document text.

    Returns: List of dicts with keys: text, position (char index)
    """
    headings = []

    if doc_type in ['md', 'txt']:
        # Markdown-style: # ## ###
        for match in re.finditer(r'^(#{1,4}\s+.+)$', text, re.MULTILINE):
            headings.append({
                "text": match.group(1).strip(),
                "position": match.start()
            })

    elif doc_type == 'pdf':
        # Heuristic: short lines, all caps, or numbered sections
        lines = text.split('\n')
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or len(stripped) > 100:
                continue
            # All caps, short
            if stripped.upper() == stripped and len(stripped.split()) <= 10:
                headings.append({"text": stripped, "position": text.find(stripped)})
            # Numbered section
            elif re.match(r'^(\d+(\.\d+)*\s+[A-Z]|Chapter\s+\d+|Section\s+\d+)', stripped):
                headings.append({"text": stripped, "position": text.find(stripped)})

    return headings


def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars per token for English."""
    return len(text) // 4


def split_by_headings(text: str, headings: List[Dict]) -> List[Dict]:
    """Split text into sections by headings."""
    if not headings:
        return [{"heading": "", "content": text, "start_pos": 0}]

    sections = []
    for i, heading in enumerate(headings):
        start = heading["position"]
        end = headings[i + 1]["position"] if i + 1 < len(headings) else len(text)
        sections.append({
            "heading": heading["text"],
            "content": text[start:end].strip(),
            "start_pos": start
        })

    return sections


def split_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs."""
    paragraphs = re.split(r'\n\n+', text)
    return [p.strip() for p in paragraphs if p.strip()]


def split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_document(text: str, doc_type: str, headings: List[Dict] = None,
                   total_pages: int = None) -> List[Dict]:
    """Split document into chunks with metadata.

    Returns: List of dicts with keys: content, page, section, chunk_index
    """
    if headings is None:
        headings = detect_headings(text, doc_type)

    MAX_CHUNK_TOKENS = 512
    OVERLAP_TOKENS = 50

    sections = split_by_headings(text, headings)
    chunks = []
    chunk_index = 0

    for section in sections:
        paragraphs = split_paragraphs(section["content"])

        for para in paragraphs:
            para_tokens = estimate_tokens(para)

            if para_tokens <= MAX_CHUNK_TOKENS:
                chunks.append({
                    "content": para,
                    "page": None,
                    "section": section["heading"],
                    "chunk_index": chunk_index,
                })
                chunk_index += 1
            else:
                # Split by sentences
                sentences = split_sentences(para)
                current_chunk = []
                current_tokens = 0

                for sentence in sentences:
                    sent_tokens = estimate_tokens(sentence)

                    if current_tokens + sent_tokens > MAX_CHUNK_TOKENS and current_chunk:
                        # Flush current chunk
                        chunk_text = " ".join(current_chunk)
                        chunks.append({
                            "content": chunk_text,
                            "page": None,
                            "section": section["heading"],
                            "chunk_index": chunk_index,
                        })
                        chunk_index += 1

                        # Start new chunk with overlap
                        overlap_text = " ".join(current_chunk)
                        overlap_sentences = []
                        overlap_len = 0
                        for s in reversed(current_chunk):
                            s_tokens = estimate_tokens(s)
                            if overlap_len + s_tokens > OVERLAP_TOKENS:
                                break
                            overlap_sentences.insert(0, s)
                            overlap_len += s_tokens

                        current_chunk = overlap_sentences + [sentence]
                        current_tokens = sum(estimate_tokens(s) for s in current_chunk)
                    else:
                        current_chunk.append(sentence)
                        current_tokens += sent_tokens

                # Flush remaining
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append({
                        "content": chunk_text,
                        "page": None,
                        "section": section["heading"],
                        "chunk_index": chunk_index,
                    })
                    chunk_index += 1

    return chunks
