from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
from resume_parser import resumeparse
from sentence_transformers import SentenceTransformer, util
import json
import spacy.cli

# Ensure en_core_web_sm model is available
try:
    import en_core_web_sm
except ImportError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    import en_core_web_sm

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
