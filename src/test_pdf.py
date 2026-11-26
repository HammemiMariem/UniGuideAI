
from pdf_processor import extract_text_from_pdf

pdf_path = "data/raw/europe/programme-guide-erasmus-27_en.pdf"

pages = extract_text_from_pdf(pdf_path)

print(f"Number of extracted pages: {len(pages)}")

for page in pages[:3]:
    print(f"\n--- Page {page['page']} ---")
    print(page["text"][:1000])
