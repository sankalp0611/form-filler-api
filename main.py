from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
import uuid
import tempfile
import traceback
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Document Processing API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
groq_client = None

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    global groq_client
    
    logger.info("🚀 Starting Document Processing API...")
    
    # Initialize Groq client
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        try:
            from groq import Groq
            groq_client = Groq(api_key=groq_api_key)
            logger.info("✅ Groq client initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Groq client: {e}")
            groq_client = None
    else:
        logger.warning("⚠️ GROQ_API_KEY not found")

@app.get("/")
async def root():
    return {"message": "Document Processing API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "groq_api_key": "present" if os.getenv("GROQ_API_KEY") else "missing",
        "groq_client": "initialized" if groq_client else "not_initialized"
    }

@app.get("/test")
async def test_endpoint():
    try:
        return {
            "message": "Test successful",
            "environment": {
                "groq_api_key": "present" if os.getenv("GROQ_API_KEY") else "missing",
                "groq_client": "ready" if groq_client else "not_ready"
            }
        }
    except Exception as e:
        logger.error(f"Test endpoint error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Test failed", "details": str(e)}
        )

def extract_text_from_file(file_path: str) -> str:
    """Extract text from various file formats"""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif file_ext == '.pdf':
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    return text
            except ImportError:
                return "PDF processing requires PyPDF2. Install with: pip install PyPDF2"
            except Exception as e:
                return f"Error processing PDF: {str(e)}"
        
        elif file_ext in ['.docx', '.doc']:
            try:
                from docx import Document
                doc = Document(file_path)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
            except ImportError:
                return "DOCX processing requires python-docx. Install with: pip install python-docx"
            except Exception as e:
                return f"Error processing DOCX: {str(e)}"
        
        else:
            return f"Unsupported file format: {file_ext}"
            
    except Exception as e:
        return f"Error reading file: {str(e)}"

async def process_with_groq(text: str, prompt: str) -> str:
    """Process text using Groq API"""
    if not groq_client:
        return f"Groq client not available. Basic processing:\n\nPrompt: {prompt}\n\nDocument content (first 500 chars):\n{text[:500]}..."
    
    try:
        # Truncate text if too long (Groq has token limits)
        max_chars = 8000  # Conservative limit
        if len(text) > max_chars:
            text = text[:max_chars] + "...\n[Content truncated due to length]"
        
        # Create the full prompt
        full_prompt = f"""
Based on the following document content, please {prompt}

Document content:
{text}

Please provide a clear and helpful response based on the document content above.
"""
        
        # Make API call to Groq
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",  # You can change this to other models
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes documents and provides insights based on the content."},
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Groq processing error: {e}")
        return f"Error processing with Groq: {str(e)}\n\nFallback - Document summary:\n{text[:500]}..."

@app.post("/process/")
async def process_document(
    file: UploadFile = File(...),
    prompt: str = Form(...)
):
    request_id = str(uuid.uuid4())
    logger.info(f"Processing request {request_id} - File: {file.filename}")
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file extension
        allowed_extensions = {'.pdf', '.docx', '.doc', '.txt'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Save file temporarily
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"{request_id}_{file.filename}")
        
        try:
            # Save uploaded file
            content = await file.read()
            with open(temp_file_path, "wb") as buffer:
                buffer.write(content)
            
            logger.info(f"File saved: {temp_file_path} ({len(content)} bytes)")
            
            # Extract text from file
            extracted_text = extract_text_from_file(temp_file_path)
            
            if not extracted_text or extracted_text.strip() == "":
                raise HTTPException(status_code=400, detail="No text could be extracted from the file")
            
            # Process with Groq
            processed_result = await process_with_groq(extracted_text, prompt)
            
            # Prepare response
            result = {
                "success": True,
                "request_id": request_id,
                "filename": file.filename,
                "file_size": len(content),
                "prompt": prompt,
                "extracted_text_length": len(extracted_text),
                "result": processed_result
            }
            
            logger.info(f"Successfully processed request {request_id}")
            return JSONResponse(content=result)
            
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    logger.info(f"Cleaned up temp file: {temp_file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup {temp_file_path}: {cleanup_error}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request {request_id}: {e}")
        logger.error(traceback.format_exc())
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "request_id": request_id,
                "error": "Processing failed",
                "details": str(e)
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
