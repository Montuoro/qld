import sys
sys.stdout.reconfigure(encoding='utf-8')

import fitz  # PyMuPDF

pdf_path = r"C:\PSAM\QLD\atar scaling\ATAR_Report_2025.pdf"
doc = fitz.open(pdf_path)

print(f"Total pages: {doc.page_count}\n")

# First pass: find which pages mention Table 6, 7, 8, 9
for i in range(doc.page_count):
    text = doc[i].get_text()
    for t in ["Table 6", "Table 7", "Table 8", "Table 9"]:
        if t in text:
            print(f"Page {i+1} contains reference to '{t}'")

print("\n" + "="*120)
print("EXTRACTING FULL TEXT FROM PAGES 10-20 (and beyond if needed)")
print("="*120 + "\n")

# Extract text from a broad range of pages to capture all tables
for i in range(8, min(25, doc.page_count)):  # pages 9-25 (0-indexed: 8-24)
    page = doc[i]
    text = page.get_text()
    print(f"\n{'='*120}")
    print(f"PAGE {i+1}")
    print(f"{'='*120}")
    print(text)

doc.close()
