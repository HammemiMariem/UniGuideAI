
def split_text(text, chunk_size=1800, overlap=300):
    """
    Split text into smaller chunks with overlapping content.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")

    chunks = []
    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        # Try to avoid cutting a word in half
        if end < len(text):
            last_space = text.rfind(" ", start, end)

            if last_space > start:
                end = last_space

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks

def create_chunks(pages, chunk_size=1800, overlap=300):
    """
    Create chunks while preserving source and page information.
    """

    all_chunks = []

    for page in pages:

        page_chunks = split_text(
            page["text"],
            chunk_size,
            overlap
        )

        for index, chunk in enumerate(page_chunks):

            all_chunks.append({
                "chunk_id": len(all_chunks) + 1,
                "source": page["source"],
                "page": page["page"],
                "chunk_index": index,
                "text": chunk
            })

    return all_chunks
