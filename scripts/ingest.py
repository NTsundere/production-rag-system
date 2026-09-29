import argparse
from pathlib import Path
from app.ingestion.loaders import load_document
from app.ingestion.chunkers import chunk_text
from app.ingestion.indexers import index_chunks
from app.logger import setup_logger

logger = setup_logger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to document")
    args = parser.parse_args()

    path = Path(args.file)
    text = load_document(path)
    chunks = chunk_text(text)
    count = index_chunks(chunks, source=path.name)
    logger.info(f"Done: {count} chunks indexed from {path.name}")