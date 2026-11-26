from embeddings import load_embedding_model

model = load_embedding_model()

print("\nModel information:")
print("Model name:", model)
print("Embedding dimension:", model.get_embedding_dimension())