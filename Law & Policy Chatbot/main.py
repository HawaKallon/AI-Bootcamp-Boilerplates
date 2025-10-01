import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dataLoading import vectorstore
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

def initialize_hf_llm():
    base_llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        temperature=0.1,
    )

    llm = ChatHuggingFace(llm=base_llm)
    return llm


llm = initialize_hf_llm()


def create_rag_chain(vectorstore, llm):
    prompt_template = """
    Use the following pieces of context to answer the question. 
    If you don’t know the answer, say "I don’t know".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        ),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa_chain


rag_chain = create_rag_chain(vectorstore, llm)


def query_rag_system(rag_chain, question):
    """Query the RAG system"""
    result = rag_chain({"query": question})

    print(f"Question: {question}")
    print(f"Answer: {result['result']}")
    print("\nSource Documents:")

    for i, doc in enumerate(result['source_documents']):
        print(f"\nSource {i + 1}:")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")


if __name__ == "__main__":
    while True:
        question = input("\nAsk a question (or type 'exit' to quit): ")
        if question.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        query_rag_system(rag_chain, question)
        print("-" * 80)


