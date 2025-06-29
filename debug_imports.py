#!/usr/bin/env python3
"""
Debug script to test all imports and dependencies
"""
import sys
import traceback

def test_basic_imports():
    print("🔍 Testing basic imports...")
    try:
        import fastapi
        print("✅ FastAPI")
    except Exception as e:
        print(f"❌ FastAPI: {e}")
    
    try:
        import uvicorn
        print("✅ Uvicorn")
    except Exception as e:
        print(f"❌ Uvicorn: {e}")

def test_ml_imports():
    print("\n🔍 Testing ML/AI imports...")
    try:
        from langchain_groq import ChatGroq
        print("✅ LangChain Groq")
    except Exception as e:
        print(f"❌ LangChain Groq: {e}")
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        print("✅ HuggingFace Embeddings (new)")
    except Exception as e:
        print(f"⚠️ HuggingFace Embeddings (new): {e}")
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            print("✅ HuggingFace Embeddings (community)")
        except Exception as e2:
            print(f"❌ HuggingFace Embeddings (community): {e2}")

def test_pdf_processing():
    print("\n🔍 Testing PDF processing imports...")
    try:
        import fitz
        print("✅ PyMuPDF (fitz)")
    except Exception as e:
        print(f"❌ PyMuPDF: {e}")
    
    try:
        import pdf2image
        print("✅ pdf2image")
    except Exception as e:
        print(f"❌ pdf2image: {e}")
    
    try:
        import pytesseract
        print("✅ pytesseract")
    except Exception as e:
        print(f"❌ pytesseract: {e}")

def test_pipeline_import():
    print("\n🔍 Testing pipeline import...")
    try:
        from pipeline import run_pipeline_on_files
        print("✅ Pipeline import successful")
        return True
    except Exception as e:
        print(f"❌ Pipeline import failed: {e}")
        traceback.print_exc()
        return False

def test_environment():
    print("\n🔍 Testing environment...")
    import os
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        print(f"✅ GROQ_API_KEY found (length: {len(groq_key)})")
    else:
        print("⚠️ GROQ_API_KEY not found")

def main():
    print("🚀 Auto Form Filler - Dependency Check\n")
    
    test_basic_imports()
    test_ml_imports()
    test_pdf_processing()
    test_environment()
    
    pipeline_ok = test_pipeline_import()
    
    print(f"\n{'='*50}")
    if pipeline_ok:
        print("✅ All critical components loaded successfully!")
        print("🚀 The application should work correctly.")
    else:
        print("❌ Pipeline import failed - check the errors above")
        print("🔧 The application may have limited functionality.")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
