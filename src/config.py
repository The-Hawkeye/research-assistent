from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "uploaded_pdfs"
CHROMA_DIR = BASE_DIR / "data" / "chroma"
COLLECTION_NAME = "docs"

DYNAMIC_CHROMA_DIR = BASE_DIR / "data" / "dynamic_chroma"
PDF_PATH = BASE_DIR / "uploaded_pdfs"

TOP_K = 5

