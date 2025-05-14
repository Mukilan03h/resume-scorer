import sys
import subprocess
import pkg_resources

# Function to check and install missing packages
def ensure_packages():
    required = {'nltk', 'sentence-transformers', 'python-multipart', 'PyPDF2', 'docx2txt'}
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = required - installed
    
    if missing:
        print(f"Installing missing packages: {missing}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])

# Ensure all packages are installed
ensure_packages()

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
import json
import nltk
import PyPDF2
import docx2txt
import re

# Ensure NLTK data is downloaded
nltk_resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']
for resource in nltk_resources:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' 
                      else f'corpora/{resource}' if resource in ['stopwords', 'words'] 
                      else f'taggers/{resource}' if resource == 'averaged_perceptron_tagger'
                      else f'chunkers/{resource}')
    except LookupError:
        print(f"Downloading NLTK resource: {resource}")
        nltk.download(resource)

from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

app = FastAPI(title="Resume Scorer API")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and fast

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def extract_resume_text(file_path):
    if file_path.lower().endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith('.docx'):
        text = extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload a PDF or DOCX file.")
    
    # Basic parsing of resume data
    parsed_data = parse_resume(text)
    return text, parsed_data

def parse_resume(text):
    # Simple parsing logic using regex and NLTK
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    
    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)
    
    # Extract skills (simplified approach)
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalpha() and word not in stop_words]
    
    # Common skills to look for
    common_skills = ['python', 'java', 'javascript', 'html', 'css', 'sql', 'react', 
                    'angular', 'node', 'aws', 'azure', 'docker', 'kubernetes',
                    'machine learning', 'data science', 'ai', 'project management']
    
    skills = [skill for skill in common_skills if skill in text.lower()]
    
    return {
        "name": "",  # Would need more complex parsing
        "email": emails[0] if emails else "",
        "phone": phones[0] if phones else "",
        "skills": skills,
        "text": text
    }

@app.post("/api/v1/score_resume")
async def score_resume(
    file: UploadFile = File(...),
    job_description: str = Form(None)
):
    suffix = os.path.splitext(file.filename)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        resume_text, parsed_data = extract_resume_text(tmp_path)

        job_score = None
        if job_description:
            embeddings = model.encode([resume_text, job_description], convert_to_tensor=True)
            job_score = float(util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()) * 100

        response = {
            "score": round(job_score if job_score is not None else 0, 2),
            "summary": "Resume scored successfully",
            "parsed_data": parsed_data
        }
    except Exception as e:
        response = {"error": str(e)}
    finally:
        os.remove(tmp_path)

    return JSONResponse(content=response)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
