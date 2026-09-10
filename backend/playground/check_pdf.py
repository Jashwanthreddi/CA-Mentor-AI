from pathlib import Path

# Use a raw string (r"...") to safely handle Windows backslashes
pdf = Path(r"C:\Users\JASHWANTHREDDY\Downloads\accounting_unit1.pdf")

print("=" * 50)
print("Checking PDF...")
print("=" * 50)

print(f"Exists : {pdf.exists()}")

if pdf.exists():
    print(f"Size   : {pdf.stat().st_size} bytes")
    
    # Read the first 20 bytes to verify the file header
    header = pdf.read_bytes()[:20]
    print(f"Header : {header}")
else:
    print("File not found. Please check the path and file name permissions.")