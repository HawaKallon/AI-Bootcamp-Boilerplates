import os
import sys
import io

sys.stderr = io.StringIO()

os.environ["USER_AGENT"] = "Mozilla/5.0 (compatible; YourBot/1.0)"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
import glob
import warnings
import logging

# ----------------------------
# CLEAN TERMINAL OUTPUT
# ----------------------------
# Suppress PyPDF2 warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("PyPDF2").setLevel(logging.ERROR)


# ----------- Load Multiple PDFs -----------

pdf_folder = "data/pdfs/"
pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))


def load_multiple_pdfs(file_paths):
    all_docs = []
    for path in file_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_docs.extend(docs)
    return all_docs


documents = load_multiple_pdfs(pdf_files)
print(f"Loaded {len(documents)} PDFs from {pdf_folder}")


# ----------- Load URLs -----------

def load_urls_from_file(file_path):
    with open(file_path, "r") as f:
        urls = [line.strip() for line in f if line.strip()]
    all_docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        docs = loader.load()
        all_docs.extend(docs)
    return all_docs


url_documents = load_urls_from_file("data/urls.txt")
print(f"Loaded {len(url_documents)} documents from URLs")

all_documents = documents + url_documents
print(f"Loaded {len(all_documents)} documents in total")


# ----------- Chunking -----------
def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)


text_chunks = chunk_documents(all_documents)
print(f"Created {len(text_chunks)} chunks")


# ----------- Embeddings -----------
def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-V2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


embeddings = create_embeddings()


# ----------- FAISS Vectorstore -----------
def create_faiss_vectorstore(chunks, embeddings):
    return FAISS.from_documents(chunks, embeddings)


vectorstore = create_faiss_vectorstore(text_chunks, embeddings)

# Save locally
vectorstore.save_local("SL_Laws_faiss")
print("✅ Vectorstore created and saved")
