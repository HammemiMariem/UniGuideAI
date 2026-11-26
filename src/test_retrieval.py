from embeddings import load_embedding_model
from vector_store import load_faiss_index, search_faiss_index
from metadata_manager import load_chunks_metadata

# Load saved data
model = load_embedding_model()
index = load_faiss_index()
chunks = load_chunks_metadata()

# Student question
question = "What is Erasmus Mundus ?"

# Generate question embedding
query_embedding = model.encode(
    "query: " + question
)

# Search for the 5 most relevant chunks
distances, indices = search_faiss_index(
    index,
    query_embedding,
    k=5
)

# Display results
print("\nRETRIEVAL RESULTS")
print("-----------------")

for rank, (distance, idx) in enumerate(
    zip(distances, indices), start=1
):
    chunk = chunks[idx]

    print(f"\nResult {rank}")
    print("Distance:", distance)
    print("Source:", chunk["source"])
    print("Page:", chunk["page"])
    print("Text:", chunk["text"][:500])