import faiss
import numpy as np


def create_faiss_index(embeddings):
    """
    Create a FAISS index from document embeddings.
    """

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    print(f"FAISS index created with {index.ntotal} vectors.")

    return index

def save_faiss_index(index, path="vectorstore/faiss_index"):
    """
    Save the FAISS index to disk.
    """

    faiss.write_index(index, path)

    print(f"FAISS index saved to: {path}")

def load_faiss_index(path="vectorstore/faiss_index"):
    """
    Load a saved FAISS index from disk.
    """

    index = faiss.read_index(path)

    print(f"FAISS index loaded with {index.ntotal} vectors.")

    return index

def search_faiss_index(index, query_embedding, k=5):
    """
    Search for the k most similar vectors in FAISS.
    """

    query_embedding = np.array(
        [query_embedding]
    ).astype("float32")

    distances, indices = index.search(query_embedding, k)

    return distances[0], indices[0]    