import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Import our custom search engine from the retriever.py file!
from retriever import get_hybrid_results

def generate_answer(query):
    """
    Takes the user's query, retrieves the best context from our hybrid search engine,
    and uses Google Gemini to synthesize a final, highly accurate answer.
    """
    # 1. Load the Gemini API Key from your .env file
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY not found. Please check your .env file.")

    # 2. Initialize the Google Gemini LLM
    # We use a low temperature (0.2) to make the AI factual and prevent hallucinations
    print("Initializing Google Gemini...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.2
    )

    # 3. Retrieve the context using the engine we built in Phase 2
    print("Running Hybrid Search Engine...")
    retrieved_docs = get_hybrid_results(query)
    
    # Combine the text of the top 3 chunks into one big string
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # 4. Create the LangChain Prompt Template
    # This strict prompt forces the LLM to act as a focused research assistant
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are an expert academic research assistant.
        Use the following pieces of retrieved context from a research paper to answer the user's question.
        
        Strict Rules:
        1. If the answer is not explicitly stated in the context, say "I cannot answer this based on the provided document." Do not guess.
        2. Keep your answer concise, analytical, and professional.
        3. Explain any technical jargon if it is defined in the context.

        Context:
        {context}

        User's Question:
        {question}

        Expert Answer:
        """
    )

    # 5. Format the prompt and generate the final answer
    print("Synthesizing answer with Gemini...\n")
    final_prompt = prompt_template.format(context=context_text, question=query)
    
    response = llm.invoke(final_prompt)
    
    return response.content

if __name__ == "__main__":
    # Ensure database exists
    if not os.path.exists("./chroma_db"):
        print("Error: Could not find './chroma_db'. Please run ingestion.py first.")
    else:
        # Ask the user for a question directly in the terminal
        print("\n" + "="*50)
        print("📚 AI Academic Research Assistant")
        print("="*50)
        
        user_query = input("Ask a question about your PDF: ")
        
        try:
            answer = generate_answer(user_query)
            
            print("\n" + "="*50)
            print("🤖 Final Synthesized Answer:")
            print("="*50)
            print(answer)
            print("\n")
            
        except Exception as e:
            print(f"\nAn error occurred: {e}")