import sys
import subprocess
import pkg_resources

# Function to check and install missing packages
def ensure_packages():
    required = {'nltk', 'resume-parser', 'sentence-transformers', 'python-multipart'}
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

# Import after ensuring packages are installed
from resume_parser import resumeparse
from sentence_transformers import SentenceTransformer, util

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

def extract_resume_text(file_path):
    data = resumeparse.read_file(file_path)
    return data.get("text", ""), data

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
