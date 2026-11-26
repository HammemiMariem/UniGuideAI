from .embeddings import load_embedding_model
from .vector_store import load_faiss_index, search_faiss_index
from .metadata_manager import load_chunks_metadata

class Retriever:

    def __init__(self):
        print("Initializing Retriever...")

        self.model = load_embedding_model()
        self.index = load_faiss_index()
        self.chunks = load_chunks_metadata()

        print("Retriever initialized successfully!")

    def retrieve(self, question, k=5):

        query_embedding = self.model.encode(
            "query: " + question
        )

        distances, indices = search_faiss_index(
            self.index,
            query_embedding,
            k
        )

        results = []

        for distance, idx in zip(distances, indices):

            results.append({
                "distance": float(distance),
                "chunk": self.chunks[idx]
            })

        return results