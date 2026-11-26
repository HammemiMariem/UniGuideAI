from pdf_processor import extract_text_from_pdf
from text_chunker import create_chunks
from embeddings import load_embedding_model

# 1. Extract PDF text
pdf_path = "data/raw/europe/programme-guide-erasmus-27_en.pdf"
pages = extract_text_from_pdf(pdf_path)

# 2. Create chunks
chunks = create_chunks(pages)

# 3. Load embedding model
model = load_embedding_model()

# 4. Generate embeddings for the first 5 chunks
texts = [
    "passage: " + chunk["text"]
    for chunk in chunks[:5]
]

embeddings = model.encode(texts)

print("Number of embeddings:", len(embeddings))
print("Embedding shape:", embeddings.shape)