import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Using standard PyPDFLoader as a reliable fallback, Unstructured for advanced layout
from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import shutil

def ingest_academic_paper(file_path):
    """
    Loads a multi-column academic paper, preserving the structural narrative.
    """
    print(f"Extracting layout from {file_path}...")
    
    # 'hi_res' strategy uses vision models to understand columns and tables
    try:
        loader = UnstructuredPDFLoader(file_path, mode="single", strategy="hi_res")
        docs = loader.load()
        print(f"Successfully loaded {len(docs)} document(s) using Unstructured.")
        return docs
    except Exception as e:
        print(f"Error loading document with unstructured: {e}")
        print("Note: unstructured requires extra dependencies like 'pdfminer.six' or 'poppler'.")
        print("Falling back to standard PyPDFLoader (which we already installed)...")
        
        # Fallback to PyPDFLoader which uses the already installed 'pypdf' package
        fallback_loader = PyPDFLoader(file_path)
        docs = fallback_loader.load()
        print(f"Successfully loaded {len(docs)} document(s) using PyPDFLoader fallback.")
        return docs

def semantically_chunk_text(docs):
    """
    Splits text based on the cosine distance of sentence embeddings.
    """
    print("Initializing embedding model for semantic chunking...")
    # Using a fast, local HuggingFace model via SentenceTransformers
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Chunking documents (this may take a moment)...")
    # Splits text when the semantic difference between sentences is in the top 20%
    text_splitter = SemanticChunker(
        embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=80
    )
    
    chunks = text_splitter.split_documents(docs)
    print(f"Generated {len(chunks)} semantic chunks.")
    
    return chunks

def build_vector_database(chunks):
    """
    Takes the semantic chunks, embeds them, and stores them in a local Chroma vector database.
    """
    print("Initializing Chroma Vector Database...")
    # We use the exact same embedding model to convert text to vectors for storage
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    persist_directory = "./chroma_db"
    
    # Clear out any old database if it exists so we start fresh
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        
    print(f"Embedding {len(chunks)} chunks and saving to {persist_directory} (this may take a minute)...")
    # Create and save the database to disk
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    print("Vector database successfully created and saved to disk!")
    return vector_db

if __name__ == "__main__":
    # --- Instructions ---
    # 1. Download a dense technical PDF (e.g., a survey paper)
    # 2. Place it in the 'data' folder
    # 3. Update the filename below to match your PDF
    
    pdf_path = "./data/sample_paper.pdf" 
    
    # Let's check both paths just in case it is in the main directory!
    if not os.path.exists(pdf_path):
        pdf_path = "./sample_paper.pdf"

    if not os.path.exists(pdf_path):
        print(f"Please put a PDF named 'sample_paper.pdf' inside the data folder, or update the path.")
    else:
        raw_document = ingest_academic_paper(pdf_path)
        processed_chunks = semantically_chunk_text(raw_document)
        
        print("\n--- Sample Chunk (First 500 characters) ---")
        print(processed_chunks[0].page_content[:500])
        print("\nPipeline Phase 1 Complete!")
        
        print("\n--- Starting Phase 2: Vector Database ---")
        vector_db = build_vector_database(processed_chunks)
        print("\nPipeline Phase 2 Complete!")