# Sierra Leone Laws System

A Retrieval-Augmented Generation (RAG) system for querying Sierra Leone legal documents using LangChain, FAISS, and Hugging Face models.

## Overview

This project implements a question-answering system that allows users to query legal documents from Sierra Leone. It uses:
- **FAISS** for efficient vector similarity search
- **HuggingFace Embeddings** for document encoding
- **Mixtral-8x7B** LLM for generating answers
- **LangChain** for orchestrating the RAG pipeline

## Features

- 📄 Load multiple PDF documents from a directory
- 🌐 Load documents from web URLs
- 🔍 Semantic search using FAISS vector store
- 🤖 AI-powered question answering with source attribution
- 💾 Persistent vector store for faster subsequent queries

## Prerequisites

- Python 3.8+
- Hugging Face API token (for Mixtral model access)

## Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

2. **Install dependencies**
```bash
pip install langchain langchain-huggingface langchain-community
pip install sentence-transformers faiss-cpu pypdf
pip install python-dotenv huggingface-hub
```

3. **Set up environment variables**

Create a `.env` file in the root directory:
```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

Get your token from: https://huggingface.co/settings/tokens

## Project Structure

```
.
├── data/
│   ├── pdfs/              # Place your PDF files here
│   └── urls.txt           # List of URLs to scrape (one per line)
├── SL_Laws_faiss/         # Generated FAISS vector store (created automatically)
├── dataLoading.py         # Data loading and vectorstore creation
├── main.py                # Main RAG application
├── .env                   # Environment variables (create this)
└── README.md              # This file
```

## Usage

### Step 1: Prepare Your Data

1. **Add PDF files**: Place your Sierra Leone legal PDFs in `data/pdfs/`

2. **Add URLs**: Create `data/urls.txt` with one URL per line:
```text
https://example.com/legal-document-1
https://example.com/legal-document-2
```

### Step 2: Create the Vector Store

Run the data loading script to process documents and create the FAISS index:

```bash
python dataLoading.py
```

This will:
- Load all PDFs from `data/pdfs/`
- Load web pages from URLs in `data/urls.txt`
- Split documents into chunks
- Generate embeddings using `sentence-transformers/all-MiniLM-L6-V2`
- Create and save a FAISS vector store to `SL_Laws_faiss/`

**Note**: You only need to run this once, or when you add new documents.

### Step 3: Query the System

Run the main application:

```bash
python main.py
```

Example interaction:
```
Ask a question (or type 'exit' to quit): How many years does a president's term last for?

Question:  How many years does a president's term last for?
Answer:  A president's term in the given context lasts for five years. This information can be found in the second schedule, where it states "No person shall hold office as President for more than two terms of five years each, whether or not the terms are consecutive."


Source Documents:

Source 1:
Content: [First 200 characters of relevant document]...
Metadata: {'source': 'data/pdfs/constitution.pdf', 'page': 5}

Source 2:
Content: [First 200 characters of relevant document]...
Metadata: {'source': 'https://example.com/legal-doc', 'page': 0}
```

Type `exit`, `quit`, or `q` to quit the application.

## Configuration

### Chunking Parameters

In `dataLoading.py`, adjust these parameters for different chunking strategies:

```python
text_chunks = chunk_documents(
    all_documents,
    chunk_size=1000,      # Size of each chunk in characters
    chunk_overlap=200     # Overlap between chunks
)
```

### Retrieval Parameters

In `main.py`, adjust the number of retrieved documents:

```python
retriever=vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Number of documents to retrieve
)
```

### LLM Temperature

Control response creativity in `main.py`:

```python
base_llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=0.1,  # Lower = more focused, Higher = more creative
)
```

## How It Works

### Data Loading Pipeline (`dataLoading.py`)

1. **Document Loading**
   - Loads PDFs using `PyPDFLoader`
   - Loads web pages using `WebBaseLoader`
   - Combines all documents into a single list

2. **Text Chunking**
   - Splits documents into smaller chunks using `RecursiveCharacterTextSplitter`
   - Maintains context with overlapping chunks

3. **Embedding Generation**
   - Converts text chunks into vector embeddings
   - Uses `sentence-transformers/all-MiniLM-L6-V2` model

4. **Vector Store Creation**
   - Creates FAISS index from embeddings
   - Saves to disk for reuse

### RAG Pipeline (`main.py`)

1. **Question Processing**
   - User asks a question
   - Question is converted to embedding

2. **Document Retrieval**
   - FAISS finds top-k most similar document chunks
   - Returns relevant context

3. **Answer Generation**
   - Combines retrieved context with question
   - Sends to Mixtral LLM via prompt template
   - Returns answer with source citations

## Troubleshooting

### Common Issues

**1. "No module named 'sentence_transformers'"**
```bash
pip install sentence-transformers
```

**2. "FAISS index not found"**
- Run `python dataLoading.py` first to create the vector store

**3. "HuggingFace API token error"**
- Ensure your `.env` file contains a valid token
- Check token permissions at https://huggingface.co/settings/tokens

**4. "Out of memory error"**
- Reduce `chunk_size` in `dataLoading.py`
- Reduce `k` (number of retrieved documents) in `main.py`

**5. Slow response times**
- First query loads the model (can take 30-60 seconds)
- Subsequent queries are faster
- Consider using smaller models for local deployment

## Performance Optimization

### For faster embedding generation:
```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-V2",
    model_kwargs={'device': 'cuda'},  # Use GPU if available
    encode_kwargs={'normalize_embeddings': True}
)
```

### For local LLM deployment:
Consider replacing the Hugging Face API with local models using:
- Ollama
- LlamaCPP
- GPT4All

## Limitations

- Answers are limited to information in the loaded documents
- Response quality depends on document quality and chunking strategy
- API rate limits may apply for Hugging Face hosted models
- Large document sets may require significant disk space for FAISS index


## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request


## Acknowledgments

- LangChain for the RAG framework
- Hugging Face for embeddings and LLM hosting
- FAISS for efficient vector search
- Sentence Transformers for embedding models


**Note**: This system is for informational purposes only and should not be considered legal advice. Always consult qualified legal professionals for legal matters.