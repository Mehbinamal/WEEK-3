from typing import List
import google.generativeai as genai
from dotenv import load_dotenv
import os
from vector_store import query_vector_store

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

class RAGSystem:
    def __init__(self):
        # Initialize Gemini model
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def query(self, user_query: str, n_results: int = 3) -> str:
        # Retrieve relevant documents from ChromaDB using vector_store
        results = query_vector_store(user_query, n_results=n_results)
        
        # Debug information
        print("\nRetrieved Results:")
        print(f"Number of chunks: {len(results['documents'][0])}")
        print("\nChunks:")
        for i, chunk in enumerate(results['documents'][0]):
            print(f"\nChunk {i+1}:")
            print(chunk)
            print(f"Distance: {results['distances'][0][i]}")
        
        # Extract retrieved documents
        retrieved_docs = results['documents'][0]
        
        # Create context from retrieved documents
        context = "\n\n".join(retrieved_docs)
        
        # Create prompt for Gemini
        prompt = f"""You are a helpful assistant that provides comprehensive answers based on the given context. 
        Please analyze ALL the provided context carefully and provide a detailed answer that:
        1. Synthesizes information from all relevant chunks
        2. Identifies the main themes and key points
        3. Provides a complete and well-structured response
        
        Context:
        {context}
        
        Question: {user_query}
        
        Please provide a comprehensive answer that considers all the information in the context:"""
        
        # Generate response using Gemini
        response = self.model.generate_content(prompt)
        
        return response.text

# Example usage
if __name__ == "__main__":
    rag = RAGSystem()
    query = "What is the main topic of the documents?"
    answer = rag.query(query)
    print(f"\nQuestion: {query}")
    print(f"Answer: {answer}")

