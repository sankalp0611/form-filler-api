from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
import uuid
import tempfile
import traceback

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

@app.get("/")
async def root():
    return {"message": "Document Processing API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "groq_api_key": "present" if os.getenv("GROQ_API_KEY") else "missing"
    }

@app.get("/test")
async def test_endpoint():
    try:
        return {
            "message": "Test successful",
            "environment": {
                "groq_api_key": "present" if os.getenv("GROQ_API_KEY") else "missing"
            }
        }
    except Exception as e:
        logger.error(f"Test endpoint error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Test failed", "details": str(e)}
        )

@app.post("/process/")
async def process_document(
    file: UploadFile = File(...),
    prompt: str = Form(...)
):
    request_id = str(uuid.uuid4())
    logger.info(f"Processing request {request_id}")
    
    try:
        # Basic validation
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Read file content
        content = await file.read()
        
        # Basic response (replace with your actual processing)
        result = {
            "success": True,
            "request_id": request_id,
            "filename": file.filename,
            "file_size": len(content),
            "prompt": prompt,
            "message": "File processed successfully"
        }
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        logger.error(traceback.format_exc())
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Processing failed",
                "details": str(e)
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
