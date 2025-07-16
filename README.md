# 🧠 DocuMind – Chat with Your PDFs using AI

DocuMind is an AI-powered conversational assistant that lets you upload PDFs and ask questions about their content. It leverages **LangChain**, **Hugging Face embeddings**, and **Groq LLMs** to create a fast, context-aware Q&A experience over documents.

---

## 🚀 Features

- 📄 Upload one or more PDFs
- 🧠 Embedding generation using Hugging Face models
- 🔎 Contextual question answering with Groq + LangChain RAG
- 🗂️ Session-based vectorstore for multi-user support
- 💬 History-aware chat
- ⚙️ FastAPI backend ready for frontend integration (React, Next.js, etc.)

---

## 🧰 Tech Stack

| Layer            | Tool/Library                   |
| ---------------- | ------------------------------ |
| Backend          | FastAPI                        |
| LLM              | Groq (Gemma2, Mixtral, etc.)   |
| Embeddings       | Hugging Face (MiniLM)          |
| Document Parsing | LangChain + PyPDFLoader        |
| Vector Storage   | ChromaDB (persistent sessions) |
| Chat Memory      | LangChain's ChatMessageHistory |
| Env Management   | python-dotenv                  |
