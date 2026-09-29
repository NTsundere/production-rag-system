from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.logger import setup_logger

logger = setup_logger(__name__)

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Семантический чанкинг с перекрытием."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    logger.info(f"Created {len(chunks)} chunks")
    return chunks