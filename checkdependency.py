import importlib
import shutil

# Python packages to check (module name : pip package name)
packages = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "multipart": "python-multipart",
    "fitz": "PyMuPDF",
    "pdf2image": "pdf2image",
    "pytesseract": "pytesseract",
    "bs4": "beautifulsoup4",
    "pdfkit": "pdfkit",
    "langchain": "langchain",
    "InstructorEmbedding": "InstructorEmbedding",
    "sentence_transformers": "sentence-transformers",
    "faiss": "faiss-cpu",
    "dotenv": "python-dotenv",
    "requests": "requests",
    "PIL": "pillow",
    "langchain_groq": "langchain-groq"
}

missing = []

print("🔍 Checking Python dependencies:\n")
for module_name, pip_name in packages.items():
    try:
        importlib.import_module(module_name)
        print(f"✅ {pip_name} — Installed")
    except ImportError:
        print(f"❌ {pip_name} — NOT Installed")
        missing.append(pip_name)

# Check external tools
print("\n🔍 Checking external tools:\n")

external_tools = {
    "tesseract": "Tesseract OCR",
    "wkhtmltopdf": "wkhtmltopdf (for pdfkit)",
    "pdftoppm": "Poppler (for pdf2image)"
}

for cmd, tool in external_tools.items():
    if shutil.which(cmd):
        print(f"✅ {tool} — Installed")
    else:
        print(f"❌ {tool} — NOT Installed or not in PATH")

# Final summary
if missing:
    print("\n⚠️ Missing Python packages:")
    print("   pip install " + " ".join(missing))
else:
    print("\n✅ All Python packages are installed.")
