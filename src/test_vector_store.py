from .pdf_processor import extract_text_from_pdf
from .text_chunker import create_chunks
from .embeddings import load_embedding_model
from .vector_store import (
    create_faiss_index,
    save_faiss_index,
    load_faiss_index
)
from .metadata_manager import (
    save_chunks_metadata,
    load_chunks_metadata
)

# Extract PDF text
pdf_path = "data/raw/europe/programme-guide-erasmus-27_en.pdf"
pages = extract_text_from_pdf(pdf_path)

# Create chunks
chunks = create_chunks(pages)
save_chunks_metadata(chunks)

# Load embedding model
model = load_embedding_model()

# Generate embeddings for all chunks
texts = [
    "passage: " + chunk["text"]
    for chunk in chunks
]

embeddings = model.encode(texts, show_progress_bar=True)

# Create FAISS index
index = create_faiss_index(embeddings)
save_faiss_index(index)

print("Total chunks:", len(chunks))
print("Total vectors:", index.ntotal)
loaded_index = load_faiss_index()

print("Loaded vectors:", loaded_index.ntotal)

loaded_chunks = load_chunks_metadata()

print("First chunk source:", loaded_chunks[0]["source"])
print("First chunk page:", loaded_chunks[0]["page"])
print("First chunk text:", loaded_chunks[0]["text"][:200])