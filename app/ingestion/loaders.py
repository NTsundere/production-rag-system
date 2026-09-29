from pathlib import Path
from pypdf import PdfReader
from app.logger import setup_logger

logger = setup_logger(__name__)

def load_document(file_path: Path) -> str:
    """Загружает текст из PDF или TXT."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix in (".txt", ".md"):
        text = file_path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported format: {suffix}")

    logger.info(f"Loaded {file_path.name}: {len(text)} chars")
    return text