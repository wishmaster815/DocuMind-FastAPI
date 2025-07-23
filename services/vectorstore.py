import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.jina import JinaEmbeddings

load_dotenv()
jina_key = os.getenv("JINA_KEY")
hf_key = os.getenv("HF_KEY")
pc_key = os.getenv("PINECONE_API_KEY")
pc_env = os.getenv("PINECONE_ENVIRONMENT")
pc_index = os.getenv("PINECONE_INDEX")

_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = JinaEmbeddings(jina_api_key=jina_key, dimension=768, model_name="jina-embeddings-v2-base-en")
    return _embeddings

pc = Pinecone(api_key=pc_key, environment=pc_env)
embeddings = get_embeddings()

if not pc.has_index(pc_index):
        pc.create_index(
        name=pc_index,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(pc_index)


# we get retrievers from this 
def generate_vectors(session_id: str,pdf_paths: list[str]):
    documents = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        documents.extend(loader.load())
    
    for path in pdf_paths:
        if os.path.exists(path):
            os.remove(path)
            print(f"[CLEANUP] Deleted file: {path}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    index_name = pc_index
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
        namespace = session_id
        )
    
    print("Embeddings successfully generated for given data")
    return vectorstore.as_retriever()

def get_retriever(session_id: str):
    return PineconeVectorStore.from_existing_index(
        index_name=pc_index,
        embedding=embeddings,
        namespace=session_id
    ).as_retriever()