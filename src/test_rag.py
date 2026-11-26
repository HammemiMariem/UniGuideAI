from rag_pipeline import RAGPipeline

# Initialize the RAG pipeline
rag = RAGPipeline()

# Ask a question
question = "What is Erasmus Mundus?"

# Generate an answer
answer = rag.generate_answer(question)

print("\nRAG ANSWER")
print("----------")
print(answer)