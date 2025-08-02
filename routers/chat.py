from fastapi import APIRouter,Form
from services.memory import get_chat_memory
from services.chain import build_conversational_chain
from services.vectorstore import get_retriever
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


router = APIRouter(prefix='/chat',tags=["chat"])

@router.post("/")
async def chat(session_id: str = Form(...), input: str = Form(...)):
    retriever = get_retriever(session_id)
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")
    rag_chain = build_conversational_chain(llm, retriever)

    # DEBUG: Check retriever output before running the chain
    docs = retriever.get_relevant_documents(input)
    if not docs:
        return {"answer": "No documents found for this session. Please upload a PDF first."}

    conversation_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: get_chat_memory(session_id),
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    result = conversation_rag_chain.invoke({"input": input}, config={"configurable": {"session_id": session_id}})
    return {"answer": result["answer"]}
