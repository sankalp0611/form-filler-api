# main.py

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import io
import re
import logging
from dotenv import load_dotenv

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Import pipeline with error handling
pipeline_available = False
try:
    from pipeline import run_pipeline_on_files
    pipeline_available = True
    logger.info("✅ Pipeline imported successfully")
except Exception as e:
    logger.error(f"❌ Failed to import pipeline: {e}")
    pipeline_available = False

@app.on_event("startup")
async def startup_event():
    """Startup checks"""
    logger.info("🚀 Starting Document Processing API...")
    logger.info(f"📁 Upload directory: {UPLOAD_DIR}")
    logger.info(f"🔧 Pipeline available: {pipeline_available}")

@app.get("/")
async def root():
    return {"message": "Document Processing API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "pipeline_available": pipeline_available,
        "upload_dir": UPLOAD_DIR
    }

def clean_llm_boilerplate(text: str) -> str:
    pattern = r"^\s*"
    return re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

@app.post("/process/")
async def process(
    pdf_file: UploadFile = File(...),
    html_file: UploadFile = File(...),
    return_pdf: bool = Form(False)
):
    try:
        # Check if pipeline is available
        if not pipeline_available:
            logger.error("Pipeline not available")
            return JSONResponse(
                status_code=500, 
                content={
                    "status": "error", 
                    "message": "Pipeline module not available. Check server logs for import errors."
                }
            )

        # Validate files
        if not pdf_file.filename or not html_file.filename:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Both PDF and HTML files are required"
                }
            )

        logger.info(f"Processing files: {pdf_file.filename}, {html_file.filename}")

        # Save uploaded files
        pdf_path = os.path.join(UPLOAD_DIR, pdf_file.filename)
        html_path = os.path.join(UPLOAD_DIR, html_file.filename)
        output_html_path = os.path.join(UPLOAD_DIR, "output.html")
        output_pdf_path = os.path.join(UPLOAD_DIR, "filled_output.pdf")

        try:
            with open(pdf_path, "wb") as f:
                shutil.copyfileobj(pdf_file.file, f)
            with open(html_path, "wb") as f:
                shutil.copyfileobj(html_file.file, f)
            
            logger.info("Files saved successfully")
        except Exception as file_error:
            logger.error(f"Error saving files: {file_error}")
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": f"Failed to save uploaded files: {str(file_error)}"
                }
            )

        # Run the agentic pipeline
        try:
            logger.info("Running pipeline...")
            result = run_pipeline_on_files(pdf_path, html_path, output_html_path, output_pdf_path)
            logger.info("Pipeline completed successfully")
        except Exception as pipeline_error:
            logger.error(f"Pipeline error: {pipeline_error}")
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": f"Pipeline processing failed: {str(pipeline_error)}"
                }
            )

        # Clean answers if needed
        cleaned_answers = {}
        try:
            for key, value in result["answers"].items():
                if isinstance(value, str):
                    cleaned_answers[key] = clean_llm_boilerplate(value)
                else:
                    cleaned_answers[key] = value  # already dict with answer and confidence
        except Exception as clean_error:
            logger.warning(f"Error cleaning answers: {clean_error}")
            cleaned_answers = result.get("answers", {})

        # Return result
        if return_pdf and os.path.exists(output_pdf_path):
            try:
                with open(output_pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                logger.info("Returning PDF file")
                return StreamingResponse(
                    io.BytesIO(pdf_bytes),
                    media_type="application/pdf",
                    headers={"Content-Disposition": "attachment; filename=filled_form.pdf"}
                )
            except Exception as pdf_error:
                logger.error(f"Error reading output PDF: {pdf_error}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "error",
                        "message": f"Failed to read output PDF: {str(pdf_error)}"
                    }
                )

        return JSONResponse(content={
            "status": "success",
            "answers": cleaned_answers,
            "tables": result.get("tables", []),
            "message": "Form filled and processed successfully."
        })

    except Exception as e:
        logger.error(f"Unexpected error in process endpoint: {e}")
        return JSONResponse(
            status_code=500, 
            content={
                "status": "error", 
                "message": f"Unexpected error: {str(e)}"
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
