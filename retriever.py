import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    """
    Manually implements the Reciprocal Rank Fusion (RRF) algorithm to combine 
    results from multiple search methods. This is the exact math used under the 
    hood by advanced search engines to merge semantic and keyword search.
    """
    fused_scores = {}

    # Calculate RRF scores for Dense (Conceptual) Results
    for rank, doc in enumerate(dense_results):
        # Use the chunk's text as a unique identifier to fuse matches
        doc_content = doc.page_content 
        if doc_content not in fused_scores:
            fused_scores[doc_content] = {"doc": doc, "score": 0.0}
        
        # The math: 1 / (rank + k) -- higher rank gets a higher fraction
        # We give dense results a slight weight multiplier (e.g., 0.5)
        fused_scores[doc_content]["score"] += 0.5 * (1 / (rank + k))

    # Calculate RRF scores for Sparse (Keyword) Results
    for rank, doc in enumerate(sparse_results):
        doc_content = doc.page_content
        if doc_content not in fused_scores:
            fused_scores[doc_content] = {"doc": doc, "score": 0.0}
            
        # We give sparse results a slight weight multiplier (e.g., 0.5)
        fused_scores[doc_content]["score"] += 0.5 * (1 / (rank + k))

    # Sort the fused dictionary by the highest combined score
    reranked_results = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
    
    # Return just the document objects in their newly ranked order
    return [item["doc"] for item in reranked_results]

def get_hybrid_results(query, persist_directory="./chroma_db"):
    """
    Executes both searches independently and fuses the results.
    """
    print("Loading database and initializing retrievers...")
    
    # 1. Setup the Dense Retriever (ChromaDB)
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    dense_retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    # 2. Extract database contents for the Sparse Retriever (BM25)
    db_data = vector_db.get()
    if not db_data['documents']:
        raise ValueError("Chroma DB is empty! Please run ingestion.py first.")
    docs = [Document(page_content=text) for text in db_data['documents']]
    
    # 3. Setup the Sparse Retriever
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 5

    # 4. Perform Independent Searches
    print(f"Searching for: '{query}'")
    dense_docs = dense_retriever.invoke(query)
    sparse_docs = bm25_retriever.invoke(query)

    # 5. Apply our custom Reciprocal Rank Fusion algorithm
    fused_docs = reciprocal_rank_fusion(dense_docs, sparse_docs)
    
    # Return the top 3 best matching chunks
    return fused_docs[:3]

if __name__ == "__main__":
    if not os.path.exists("./chroma_db"):
        print("Could not find './chroma_db'. Please run ingestion.py first to build the database.")
    else:
        # Define our complex query
        query = "What are the key techniques used for instruction tuning a pre-trained model?"
        
        # Perform the hybrid search using our custom engine
        results = get_hybrid_results(query)
        
        # Display the results
        print(f"\n--- Top {len(results)} Retrieved Chunks ---")
        for i, doc in enumerate(results):
            print(f"\n[Result #{i+1}]")
            print(doc.page_content[:400] + "...\n")
            print("-" * 50)