from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"


def load_embedding_model():
    print("Loading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    print("Embedding model loaded successfully!")

    return model