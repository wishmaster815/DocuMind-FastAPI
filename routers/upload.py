from fastapi import APIRouter,UploadFile,File
from services.vectorstore import generate_vectors
import os, uuid

router = APIRouter(prefix="/upload", tags=["upload"])

UPLOAD_DIR = "uploads/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/")
async def upload_pdfs(files:list[UploadFile] = File(...)):
    session_id = str(uuid.uuid4())
    file_paths = []

    for file in files:
        path = os.path.join(UPLOAD_DIR,f"{session_id}_{file.filename}")
        with open(path,'wb') as output:
            output.write(await file.read())
            file_paths.append(path)

    docs = generate_vectors(session_id,file_paths)

    return {
        'session_id':session_id,
        'message':'pdfs processed successfully!'
    }
