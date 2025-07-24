from fastapi import FastAPI
from routers import upload, chat
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["http://localhost:3000", "https://documind-ai-815.vercel.app/"]
    
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(upload.router)
app.include_router(chat.router)

@app.get("/")
def root():
    return {"message": "DocuMind API is live!"}