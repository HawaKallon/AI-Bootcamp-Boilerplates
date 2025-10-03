import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # API Settings
    API_TITLE = "Sierra Leone Legal Assistant API"
    API_DESCRIPTION = "API for querying Sierra Leone laws and policies"
    API_VERSION = "1.0.0"

    # CORS Settings
    CORS_ORIGINS = ["*"]  # In production, specify your frontend URL

    # Hugging Face Settings
    HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    HF_MODEL_REPO = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    HF_TEMPERATURE = 0.1
    HF_MAX_NEW_TOKENS = 512

    # Embeddings Settings
    EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDINGS_DEVICE = "cpu"

    # Vector Store Settings
    VECTORSTORE_PATH = "SL_Laws_faiss"

    # RAG Settings
    RETRIEVAL_K = 4  # Number of documents to retrieve


config = Config()