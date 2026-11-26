
import pymupdf
from pathlib import Path


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF while preserving page numbers.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    extracted_pages = []

    with pymupdf.open(pdf_path) as document:

        for page_number, page in enumerate(document, start=1):

            text = page.get_text("text")

            if text.strip():
                extracted_pages.append({
                    "page": page_number,
                    "text": text.strip(),
                    "source": pdf_path.name
                })

    return extracted_pages


if __name__ == "__main__":
    print("PDF Processor module is ready!")
