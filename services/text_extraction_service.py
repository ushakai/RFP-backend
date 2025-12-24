"""
Production-grade text extraction service
Handles PDF, DOCX, TXT, MD with quality checks and error handling
"""
import io
import os
import re
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# PDF extraction
try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️  PyPDF2 not installed, PDF extraction unavailable")

# DOCX extraction
try:
    import docx2txt
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("⚠️  docx2txt not installed, DOCX extraction unavailable")

from config.rag_config import (
    SUPPORTED_TEXT_FORMATS,
    MAX_TEXT_FILE_SIZE_MB,
    ENABLE_OCR,
    MIN_TEXT_QUALITY_SCORE,
    RAG_DEBUG_LOGGING,
)


# ==================== FILE TYPE DETECTION ====================

def detect_file_type(filename: str, content_type: Optional[str] = None) -> str:
    """
    Detect file type from filename and optional MIME type
    
    Returns: Extension without dot (e.g., 'pdf', 'docx', 'txt')
    """
    # Try from filename first
    ext = Path(filename).suffix.lower().lstrip('.')
    
    # Validate extension
    if ext in ['txt', 'md', 'markdown', 'pdf', 'docx', 'doc']:
        return ext
    
    # Try from content type
    if content_type:
        mime_map = {
            'text/plain': 'txt',
            'text/markdown': 'md',
            'application/pdf': 'pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
            'application/msword': 'doc',
        }
        if content_type in mime_map:
            return mime_map[content_type]
    
    # Default to txt if cannot determine
    return 'txt'


def is_supported_format(file_type: str) -> bool:
    """Check if file format is supported"""
    return file_type in SUPPORTED_TEXT_FORMATS and SUPPORTED_TEXT_FORMATS[file_type]


# ==================== TEXT EXTRACTION ====================

def extract_text_from_txt(content: bytes) -> str:
    """Extract text from TXT file"""
    try:
        # Try UTF-8 first
        return content.decode('utf-8')
    except UnicodeDecodeError:
        try:
            # Fallback to Latin-1
            return content.decode('latin-1')
        except Exception as e:
            raise ValueError(f"Failed to decode text file: {e}")


def extract_text_from_md(content: bytes) -> str:
    """Extract text from Markdown file (same as TXT but preserve structure)"""
    text = extract_text_from_txt(content)
    
    # Remove markdown syntax for cleaner text (optional)
    # Keep it for now to preserve structure
    return text


def extract_text_from_pdf(content: bytes) -> str:
    """Extract text from PDF file"""
    if not PDF_AVAILABLE:
        raise RuntimeError("PDF extraction not available. Install PyPDF2.")
    
    try:
        pdf_file = io.BytesIO(content)
        reader = PdfReader(pdf_file)
        
        text_parts = []
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_parts.append(page_text)
            except Exception as e:
                if RAG_DEBUG_LOGGING:
                    print(f"   Warning: Failed to extract page {page_num + 1}: {e}")
                continue
        
        if not text_parts:
            # Try OCR if enabled and no text found
            if ENABLE_OCR:
                return extract_text_with_ocr(content, 'pdf')
            else:
                raise ValueError("No text found in PDF. Enable OCR for scanned PDFs.")
        
        full_text = '\n\n'.join(text_parts)
        return full_text
        
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {e}")


def extract_text_from_docx(content: bytes) -> str:
    """Extract text from DOCX file"""
    if not DOCX_AVAILABLE:
        raise RuntimeError("DOCX extraction not available. Install docx2txt.")
    
    try:
        # docx2txt expects a file path or file-like object
        docx_file = io.BytesIO(content)
        text = docx2txt.process(docx_file)
        
        if not text or not text.strip():
            raise ValueError("No text found in DOCX file")
        
        return text
        
    except Exception as e:
        raise ValueError(f"Failed to extract text from DOCX: {e}")


def extract_text_with_ocr(content: bytes, file_type: str) -> str:
    """
    Extract text using OCR (for scanned PDFs/images)
    Requires pytesseract and Pillow
    """
    if not ENABLE_OCR:
        raise RuntimeError("OCR is not enabled. Set RAG_ENABLE_OCR=1 in .env")
    
    try:
        import pytesseract
        from PIL import Image
        from pdf2image import convert_from_bytes
    except ImportError:
        raise RuntimeError(
            "OCR dependencies not installed. "
            "Install: pip install pytesseract pillow pdf2image"
        )
    
    try:
        if file_type == 'pdf':
            # Convert PDF pages to images
            images = convert_from_bytes(content)
            
            text_parts = []
            for i, image in enumerate(images):
                if RAG_DEBUG_LOGGING:
                    print(f"   OCR processing page {i + 1}...")
                page_text = pytesseract.image_to_string(image)
                if page_text and page_text.strip():
                    text_parts.append(page_text)
            
            return '\n\n'.join(text_parts)
        else:
            # Direct image OCR
            image = Image.open(io.BytesIO(content))
            return pytesseract.image_to_string(image)
            
    except Exception as e:
        raise ValueError(f"OCR extraction failed: {e}")


def extract_text(
    content: bytes,
    filename: str,
    content_type: Optional[str] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Main text extraction function
    
    Args:
        content: File content as bytes
        filename: Original filename
        content_type: Optional MIME type
        
    Returns:
        Tuple of (extracted_text, metadata)
        
    Raises:
        ValueError: If extraction fails or file is unsupported
    """
    # Validate file size
    file_size_mb = len(content) / (1024 * 1024)
    if file_size_mb > MAX_TEXT_FILE_SIZE_MB:
        raise ValueError(
            f"File too large: {file_size_mb:.1f}MB (max: {MAX_TEXT_FILE_SIZE_MB}MB)"
        )
    
    # Detect file type
    file_type = detect_file_type(filename, content_type)
    
    if not is_supported_format(file_type):
        raise ValueError(
            f"Unsupported file format: {file_type}. "
            f"Supported: {', '.join([k for k, v in SUPPORTED_TEXT_FORMATS.items() if v])}"
        )
    
    # Extract based on file type
    metadata = {
        'filename': filename,
        'file_type': file_type,
        'file_size_bytes': len(content),
        'file_size_mb': file_size_mb,
    }
    
    if RAG_DEBUG_LOGGING:
        print(f"   Extracting text from {file_type.upper()} file ({file_size_mb:.2f}MB)...")
    
    try:
        if file_type == 'txt':
            text = extract_text_from_txt(content)
        elif file_type in ['md', 'markdown']:
            text = extract_text_from_md(content)
        elif file_type == 'pdf':
            text = extract_text_from_pdf(content)
        elif file_type == 'docx':
            text = extract_text_from_docx(content)
        elif file_type == 'doc':
            # DOC requires different library (antiword or LibreOffice)
            raise NotImplementedError(
                "Legacy DOC format not supported. Convert to DOCX first."
            )
        else:
            raise ValueError(f"No extractor for file type: {file_type}")
        
        # Clean and validate extracted text
        text = clean_extracted_text(text)
        
        if not text or len(text.strip()) < 10:
            raise ValueError("Extracted text is too short or empty")
        
        # Add extraction metadata
        metadata['extraction_method'] = file_type
        metadata['char_count'] = len(text)
        metadata['word_count'] = len(text.split())
        metadata['line_count'] = len(text.split('\n'))
        
        if RAG_DEBUG_LOGGING:
            print(f"   ✓ Extracted {metadata['char_count']:,} chars, {metadata['word_count']:,} words")
        
        return text, metadata
        
    except Exception as e:
        error_msg = f"Text extraction failed for {file_type}: {str(e)}"
        if RAG_DEBUG_LOGGING:
            import traceback
            traceback.print_exc()
        raise ValueError(error_msg)


# ==================== TEXT CLEANING ====================

def clean_extracted_text(text: str) -> str:
    """
    Clean extracted text while preserving structure
    
    Removes:
    - Excessive whitespace
    - Control characters
    - Page headers/footers artifacts
    
    Preserves:
    - Paragraph structure
    - Sentence boundaries
    - Line breaks (meaningful ones)
    """
    if not text:
        return ""
    
    # Remove null bytes and other control characters
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Normalize Unicode
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    
    # Remove page numbers and common header/footer patterns
    # Pattern: "Page 1 of 10", "Page 1", "1/10", etc.
    text = re.sub(r'\n\s*Page\s+\d+(\s+of\s+\d+)?\s*\n', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'\n\s*\d+\s*/\s*\d+\s*\n', '\n', text)
    text = re.sub(r'\n\s*\[\s*\d+\s*\]\s*\n', '\n', text)  # [1], [2], etc.
    
    # Collapse excessive newlines (max 2 consecutive)
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    
    # Collapse excessive spaces (max 2 consecutive)
    text = re.sub(r' {3,}', '  ', text)
    
    # Fix common OCR errors
    text = fix_ocr_errors(text)
    
    # Remove trailing/leading whitespace from each line
    lines = text.split('\n')
    lines = [line.rstrip() for line in lines]
    text = '\n'.join(lines)
    
    # Final cleanup
    text = text.strip()
    
    return text


def fix_ocr_errors(text: str) -> str:
    """Fix common OCR errors"""
    # Common OCR mistakes
    replacements = {
        r'\bl\b': 'I',  # Lowercase l mistaken for I
        r'\bO\b': '0',  # Uppercase O mistaken for zero in numbers
        r'\b0\b': 'O',  # Zero mistaken for O in words
        r'(\w)-\s*\n\s*(\w)': r'\1\2',  # Remove hyphenation at line breaks
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    
    return text


# ==================== TEXT QUALITY ASSESSMENT ====================

def assess_extraction_quality(text: str, metadata: Dict[str, Any]) -> float:
    """
    Assess quality of extracted text (0-1 score)
    
    Checks:
    - Character distribution
    - Word statistics
    - Structure indicators
    - Readability
    """
    if not text or len(text) < 10:
        return 0.0
    
    quality_score = 1.0
    
    # 1. Character distribution
    alpha_chars = sum(1 for c in text if c.isalpha())
    digit_chars = sum(1 for c in text if c.isdigit())
    space_chars = sum(1 for c in text if c.isspace())
    punct_chars = sum(1 for c in text if c in '.,!?;:\'"-()[]{}')
    total_chars = len(text)
    
    alpha_ratio = alpha_chars / total_chars
    
    # Text should be mostly alphabetic
    if alpha_ratio < 0.5:
        quality_score *= 0.7  # Penalize non-textual content
    
    # 2. Word statistics
    words = text.split()
    if words:
        avg_word_len = sum(len(w) for w in words) / len(words)
        
        # Average word length should be 3-12 characters
        if avg_word_len < 2 or avg_word_len > 20:
            quality_score *= 0.8
    
    # 3. Check for sentence structure
    has_punctuation = any(c in text for c in '.!?')
    if not has_punctuation and len(text) > 100:
        quality_score *= 0.9  # Penalize lack of sentence structure
    
    # 4. Check for excessive special characters (corrupted extraction)
    special_chars = total_chars - (alpha_chars + digit_chars + space_chars + punct_chars)
    special_ratio = special_chars / total_chars
    
    if special_ratio > 0.2:
        quality_score *= 0.5  # Heavy penalty for corrupted text
    
    # 5. Check extraction method reliability
    if metadata.get('extraction_method') == 'ocr':
        quality_score *= 0.95  # OCR is slightly less reliable
    
    return min(1.0, max(0.0, quality_score))


def validate_extracted_text(text: str, metadata: Dict[str, Any], min_quality: float = MIN_TEXT_QUALITY_SCORE) -> bool:
    """
    Validate extracted text meets quality threshold
    
    Returns:
        True if text is good quality, False otherwise
    """
    quality = assess_extraction_quality(text, metadata)
    metadata['quality_score'] = quality
    
    if quality < min_quality:
        if RAG_DEBUG_LOGGING:
            print(f"   ⚠️  Low quality extraction: {quality:.2f} < {min_quality}")
        return False
    
    if RAG_DEBUG_LOGGING:
        print(f"   ✓ Quality score: {quality:.2f}")
    
    return True


# ==================== DIAGNOSTICS ====================

def get_extraction_capabilities() -> Dict[str, Any]:
    """Get available extraction capabilities"""
    return {
        'pdf': PDF_AVAILABLE,
        'docx': DOCX_AVAILABLE,
        'txt': True,
        'markdown': True,
        'ocr': ENABLE_OCR,
        'max_file_size_mb': MAX_TEXT_FILE_SIZE_MB,
        'supported_formats': [k for k, v in SUPPORTED_TEXT_FORMATS.items() if v],
    }


def test_extraction():
    """Test extraction with sample files"""
    print("🔍 Testing text extraction service...")
    
    capabilities = get_extraction_capabilities()
    print(f"\n📋 Capabilities:")
    for fmt, available in capabilities.items():
        status = "✅" if available else "❌"
        print(f"   {status} {fmt}")
    
    # Test with simple text
    test_content = b"This is a test document.\n\nIt has multiple paragraphs.\n\nFor testing purposes."
    
    try:
        text, metadata = extract_text(test_content, "test.txt")
        print(f"\n✓ TXT extraction successful:")
        print(f"   Chars: {metadata['char_count']}")
        print(f"   Words: {metadata['word_count']}")
        print(f"   Quality: {assess_extraction_quality(text, metadata):.2f}")
        
        return True
    except Exception as e:
        print(f"\n❌ Extraction test failed: {e}")
        return False


if __name__ == "__main__":
    test_extraction()

