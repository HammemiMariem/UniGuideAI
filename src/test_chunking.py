
from pdf_processor import extract_text_from_pdf
from text_chunker import create_chunks

pdf_path = "data/raw/europe/programme-guide-erasmus-27_en.pdf"

# Step 1: Extract PDF text
pages = extract_text_from_pdf(pdf_path)

# Step 2: Create chunks
chunks = create_chunks(pages)

# Step 3: Display statistics
print("TEXT CHUNKING RESULTS")
print("---------------------")

print("Extracted pages:", len(pages))
print("Generated chunks:", len(chunks))

# Step 4: Display the first chunk
if chunks:
    print("\nFIRST CHUNK")
    print("Chunk ID:", chunks[0]["chunk_id"])
    print("Source:", chunks[0]["source"])
    print("Page:", chunks[0]["page"])
    print("Text:")
    print(chunks[0]["text"])
