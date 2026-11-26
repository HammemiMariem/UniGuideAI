from retriever import Retriever

# Initialize the retriever
retriever = Retriever()

# Ask a question
question = "What is Erasmus Mundus?"

# Retrieve relevant information
results = retriever.retrieve(question, k=5)

# Display results
print("\nRETRIEVED INFORMATION")
print("---------------------")

for rank, result in enumerate(results, start=1):

    chunk = result["chunk"]

    print(f"\nResult {rank}")
    print("Distance:", result["distance"])
    print("Source:", chunk["source"])
    print("Page:", chunk["page"])
    print("Text:", chunk["text"][:500])