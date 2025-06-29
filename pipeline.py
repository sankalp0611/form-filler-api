import os
import json
import fitz
from bs4 import BeautifulSoup
from pdf2image import convert_from_path
import pytesseract
import pdfkit
import platform
import shutil
import tempfile
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# === ✅ Fix paths for Linux deployment ===
if platform.system() == 'Windows':
    poppler_path = r"C:\\poppler\\poppler-24.08.0\\Library\\bin"
    wkhtmltopdf_path = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
else:
    # Linux (Render) - use system PATH
    poppler_path = None
    wkhtmltopdf_path = shutil.which("wkhtmltopdf")

if wkhtmltopdf_path is None:
    raise EnvironmentError("wkhtmltopdf executable not found. Make sure it's installed and accessible in PATH.")

pdfkit_config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)


class PdfOcrExtractorNode:
    def __init__(self, poppler_path):
        self.poppler_path = poppler_path

    def run(self, pdf_path):
        if self.poppler_path:
            images = convert_from_path(pdf_path, poppler_path=self.poppler_path)
        else:
            images = convert_from_path(pdf_path)
        text = "\n".join(pytesseract.image_to_string(img) for img in images).strip()
        return text if text else "[OCR Failed: No text extracted]"


class HtmlFormExtractorNode:
    def run(self, html_path):
        with open(html_path, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")
        inputs = soup.find_all(["input", "textarea", "select"])
        return [
            (tag.get("name") or tag.get("id") or tag.get("placeholder") or "Unknown field").strip()
            for tag in inputs
        ]


class GroqQAConfidenceNode:
    def __init__(self):
        self.llm = ChatGroq(temperature=0.2, model_name="llama3-8b-8192")
        self.qa_prompt = PromptTemplate(
            input_variables=["doc_text", "question"],
            template="""
You are an intelligent document assistant.
Based on the extracted document text below, answer the question as accurately as possible.
If unsure, say "Not found".

Document Text:
{doc_text}

Question:
{question}

Answer:
"""
        )
        self.confidence_prompt = PromptTemplate(
            input_variables=["doc_text", "question", "answer"],
            template="""
You previously answered the following question based on the document:

Document Text:
{doc_text}

Question:
{question}

Your Answer:
{answer}

Now, rate your confidence in this answer on a scale of 1 (low) to 5 (high).
Just return the number.
"""
        )
        self.qa_chain = self.qa_prompt | self.llm
        self.confidence_chain = self.confidence_prompt | self.llm

    def run(self, doc_text, questions):
        results = {}
        for q in questions:
            answer = self.qa_chain.invoke({"doc_text": doc_text, "question": q}).content.strip()
            confidence = self.confidence_chain.invoke({
                "doc_text": doc_text,
                "question": q,
                "answer": answer
            }).content.strip()
            results[q] = {"answer": answer, "confidence": confidence}
        return results


class GroqTableExtractorNode:
    def __init__(self):
        self.llm = ChatGroq(model_name="llama3-8b-8192")

    def run(self, doc_text):
        prompt = f"""
You are a document understanding AI.
Extract any tabular data and return as JSON list of lists (each list is a row).

Document Text:
{doc_text}

Structured Table:
"""
        response = self.llm.invoke(prompt).content.strip()
        try:
            return json.loads(response) if "No table" not in response else []
        except Exception:
            return []


class PdfEmbeddingSearchNode:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.llm = ChatGroq(model_name="llama3-8b-8192")
        self.rerank_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""
You are a helpful assistant. Based on the context below, answer the question.

Context:
{context}

Question:
{question}

Return your answer and a confidence score from 1 to 5.
"""
        )
        self.rerank_chain = self.rerank_prompt | self.llm
        self.vector_store = None

    def build_vector_index(self, pdf_path, chunk_size=500):
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        # Use temporary directory for Render
        temp_dir = tempfile.mkdtemp()
        self.vector_store = Chroma.from_texts(chunks, self.embedding_model, persist_directory=temp_dir)

    def answer_question(self, question, k=3):
        docs = self.vector_store.similarity_search(question, k=k)
        context = "\n".join([doc.page_content for doc in docs])
        return self.rerank_chain.invoke({"question": question, "context": context}).content.strip()


class PdfOutputWriterNode:
    def __init__(self, pdfkit_config):
        self.pdfkit_config = pdfkit_config

    def save_answers_as_clean_pdf(self, answers_map, output_html_path, output_pdf_path):
        html = """
        <html><head><style>
        body { font-family: Arial; margin: 40px; }
        .q-block { margin-bottom: 20px; }
        .confidence.high { color: green; }
        .confidence.low { color: red; }
        </style></head><body><h1>Filled Form</h1>
        """
        for q, data in answers_map.items():
            conf_class = "high" if data['confidence'].isdigit() and int(data['confidence']) >= 3 else "low"
            html += f"""
            <div class='q-block'>
                <p><strong>Q:</strong> {q}</p>
                <p><strong>A:</strong> {data['answer']}</p>
                <p class='confidence {conf_class}'><strong>Confidence:</strong> {data['confidence']}</p>
            </div>
            """
        html += "</body></html>"

        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(html)
        pdfkit.from_file(output_html_path, output_pdf_path, configuration=self.pdfkit_config)


class AgenticDocumentProcessor:
    def __init__(self, pdf_path, html_path, output_html_path, output_pdf_path):
        self.pdf_path = pdf_path
        self.html_path = html_path
        self.output_html_path = output_html_path
        self.output_pdf_path = output_pdf_path

        self.ocr_node = PdfOcrExtractorNode(poppler_path)
        self.form_extractor_node = HtmlFormExtractorNode()
        self.qa_node = GroqQAConfidenceNode()
        self.table_node = GroqTableExtractorNode()
        self.embedding_node = PdfEmbeddingSearchNode()
        self.pdf_writer_node = PdfOutputWriterNode(pdfkit_config)

    def run_full_pipeline(self):
        doc_text = self.ocr_node.run(self.pdf_path)
        questions = self.form_extractor_node.run(self.html_path)
        answers_map = self.qa_node.run(doc_text, questions)
        self.embedding_node.build_vector_index(self.pdf_path)

        # === Phase 5 fallback rerank ===
        for q, result in answers_map.items():
            if result['confidence'].isdigit() and int(result['confidence']) <= 2:
                fallback = self.embedding_node.answer_question(q)
                result['answer'] = f"{result['answer']} [fallback: {fallback}]"

        tables = self.table_node.run(doc_text)

        # Output writing logic
        self.pdf_writer_node.save_answers_as_clean_pdf(answers_map, self.output_html_path, self.output_pdf_path)

        return {
            "answers": answers_map,
            "tables": tables,
            "output_pdf_path": self.output_pdf_path
        }


def run_pipeline_on_files(pdf_path, html_path, output_html_path, output_pdf_path):
    processor = AgenticDocumentProcessor(pdf_path, html_path, output_html_path, output_pdf_path)
    return processor.run_full_pipeline()
