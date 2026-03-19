
import fitz

def extract_pdf_text_by_page(path):
    doc = fitz.open(str(path))
    return [{"page": i, "text": p.get_text()} for i, p in enumerate(doc)]

def chunk_text(text, size=1000):
    return [text[i:i+size] for i in range(0, len(text), size)]
