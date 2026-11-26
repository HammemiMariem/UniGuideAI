import json
from pathlib import Path


def save_chunks_metadata(chunks, path="vectorstore/chunks_metadata.json"):
    """
    Save text chunks and their metadata as a JSON file.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=4)

    print(f"Chunks metadata saved to: {path}")
def load_chunks_metadata(path="vectorstore/chunks_metadata.json"):
    """
    Load saved text chunks and their metadata.
    """

    with open(path, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    print(f"Loaded {len(chunks)} chunks.")

    return chunks    
    