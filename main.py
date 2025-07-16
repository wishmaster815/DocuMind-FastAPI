from fastapi import FastAPI
from routers import upload, chat

app = FastAPI()

app.include_router(upload.router)
app.include_router(chat.router)

@app.get("/")
def root():
    return {"message": "DocuMind API is live!"}