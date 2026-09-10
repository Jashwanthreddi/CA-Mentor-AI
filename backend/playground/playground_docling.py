"""
Docling Playground

Purpose:
---------
Understand the structure returned by Docling before
building the production extractor.
"""

from pathlib import Path
from pprint import pprint

from docling.document_converter import DocumentConverter

# PDF Path
pdf_path = Path("C:/Users/JASHWANTHREDDY/Downloads/accounting_unit1.pdf")

print("=" * 60)
print("Loading PDF...")
print("=" * 60)

converter = DocumentConverter()

result = converter.convert(pdf_path)

print("\nConversion Successful ✅")

print("\nResult Type:")
print(type(result))

print("\nAvailable Attributes:")
pprint(dir(result))