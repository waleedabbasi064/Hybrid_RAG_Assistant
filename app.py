import streamlit as st
import os

# Import the exact function you just tested in the terminal!
from generator import generate_answer
# Import our pipeline functions to run them on the fly!
from python_ingestion import ingest_academic_paper, semantically_chunk_text, build_vector_database

# Set up the look of the webpage
st.set_page_config(page_title="Academic PDF Assistant", page_icon="📚", layout="centered")

st.title("📚 AI Academic Research Assistant")

# --- NEW: Sidebar for Document Upload ---
with st.sidebar:
    st.header("📄 Document Management")
    uploaded_file = st.file_uploader("Upload a new PDF paper", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Process New Document"):
            with st.spinner("Processing PDF (Extracting, Chunking, Embedding)... This may take a minute."):
                # Save the uploaded file temporarily so our loader can read it
                temp_pdf_path = "temp_uploaded.pdf"
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    # Run our ingestion pipeline functions!
                    raw_document = ingest_academic_paper(temp_pdf_path)
                    processed_chunks = semantically_chunk_text(raw_document)
                    build_vector_database(processed_chunks)
                    
                    st.success("Document processed successfully!")
                    
                    # Clear old chat history since we have a new document
                    st.session_state.messages = [] 
                except Exception as e:
                    st.error(f"Error processing document: {e}")
                finally:
                    # Clean up the temporary file
                    if os.path.exists(temp_pdf_path):
                        os.remove(temp_pdf_path)
# ----------------------------------------

# Safety check: ensure the database was built
if not os.path.exists("./chroma_db"):
    st.info("👋 Welcome! Please upload and process a PDF document from the sidebar to get started.")
    st.stop() # Stops the app from running further until the DB exists

st.markdown("Welcome! I have read your academic paper. What would you like to know?")

# Initialize a "session state" to remember the chat history while the user is on the page
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all previous chat messages on the screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create the chat input box at the bottom of the screen
if user_query := st.chat_input("Ask a question about your PDF..."):
    
    # 1. Show the user's message on the screen immediately
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Add user message to our saved chat history
    st.session_state.messages.append({"role": "user", "content": user_query})

    # 2. Show a loading spinner while our AI does the heavy lifting
    with st.chat_message("assistant"):
        with st.spinner("Searching the database and synthesizing answer..."):
            try:
                # Call our Hybrid RAG pipeline!
                answer = generate_answer(user_query)
                st.markdown(answer)
                
                # Add the AI's answer to our saved chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"An error occurred: {e}")