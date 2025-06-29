# main.py

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import io
import re
from dotenv import load_dotenv
from pipeline import run_pipeline_on_files

load_dotenv()

app = FastAPI(
    title="Auto Form Filler API",
    description="API for processing PDF documents and filling HTML forms using AI",
    version="1.0.0"
)

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

def clean_llm_boilerplate(text: str) -> str:
    pattern = r"^\s*"
    return re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

# ✅ ADD ROOT ROUTE - This fixes the 404 error!
@app.get("/")
async def root():
    return {
        "message": "Auto Form Filler API is running!",
        "status": "healthy",
        "endpoints": {
            "process": "/process/ (POST)",
            "docs": "/docs",
            "health": "/health"
        }
    }

# ✅ ADD HEALTH CHECK ENDPOINT
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "auto-form-filler"}

@app.post("/process/")
async def process(
    pdf_file: UploadFile = File(...),
    html_file: UploadFile = File(...),
    return_pdf: bool = Form(False)
):
    try:
        # Save uploaded files
        pdf_path = os.path.join(UPLOAD_DIR, pdf_file.filename)
        html_path = os.path.join(UPLOAD_DIR, html_file.filename)
        output_html_path = os.path.join(UPLOAD_DIR, "output.html")
        output_pdf_path = os.path.join(UPLOAD_DIR, "filled_output.pdf")

        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(pdf_file.file, f)
        with open(html_path, "wb") as f:
            shutil.copyfileobj(html_file.file, f)

        # Run the agentic pipeline
        result = run_pipeline_on_files(pdf_path, html_path, output_html_path, output_pdf_path)

        # Clean answers if needed
        cleaned_answers = {}
        for key, value in result["answers"].items():
            if isinstance(value, str):
                cleaned_answers[key] = clean_llm_boilerplate(value)
            else:
                cleaned_answers[key] = value  # already dict with answer and confidence

        # Return result
        if return_pdf and os.path.exists(output_pdf_path):
            with open(output_pdf_path, "rb") as f:
                pdf_bytes = f.read()
            return StreamingResponse(
                io.BytesIO(pdf_bytes),
                media_type="application/pdf",
                headers={"Content-Disposition": "attachment; filename=filled_form.pdf"}
            )

        return JSONResponse(content={
            "status": "success",
            "answers": cleaned_answers,
            "tables": result["tables"],
            "message": "Form filled and processed successfully."
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
