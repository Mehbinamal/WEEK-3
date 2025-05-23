from typing import List
import pymupdf  
from langchain.docstore.document import Document
import chromadb
from chromadb.config import Settings


def chunk_text(text: str) -> List[Document]:
    """
    Create one chunk per line of text.
    
    Args:
        text (str): The text to chunk
        
    Returns:
        List[Document]: List of Document objects, one per line
    """
    # Split text into lines and create a chunk for each non-empty line
    chunks = []
    for i, line in enumerate(text.split('\n')):
        line = line.strip()
        if line:  # Only create chunks for non-empty lines
            chunks.append(Document(
                page_content=line,
                metadata={"source": "local", "chunk_index": i}
            ))
    
    return chunks

#function to extract text from pdf
def extract_text_from_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)   
    all_text = ""  
    for page in doc:
        all_text += page.get_text()
    doc.close()
    return all_text

def create_vector_store(chunks: List[Document], collection_name: str = "documents") -> chromadb.Collection:
    """
    Create and populate a ChromaDB collection with the document chunks.
    
    Args:
        chunks (List[Document]): List of document chunks to store
        collection_name (str): Name of the ChromaDB collection
        
    Returns:
        chromadb.Collection: The created ChromaDB collection
    """
    try:
        # Initialize ChromaDB client
        client = chromadb.Client(Settings(
            persist_directory="./chroma_db",
            is_persistent=True
        ))
        
        # Create or get collection
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Prepare documents for storage
        documents = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [f"doc_{i}" for i in range(len(chunks))]
        
        # Add documents to collection
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        return collection
        
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        raise

def process_and_store_document(text: str, collection_name: str = "documents") -> chromadb.Collection:
    """
    Process text into chunks and store them in ChromaDB.
    
    Args:
        text (str): The text to process and store
        collection_name (str): Name of the ChromaDB collection
        
    Returns:
        chromadb.Collection: The ChromaDB collection containing the chunks
    """
    # First chunk the text
    chunks = chunk_text(text)
    
    # Then store in ChromaDB
    collection = create_vector_store(chunks, collection_name)
    
    return collection

def query_vector_store(query: str, n_results: int = 5, collection: chromadb.Collection = None) -> dict:
    """
    Query the vector store for similar documents.
    
    Args:
        query (str): The search query
        n_results (int): Number of results to return
        collection (chromadb.Collection, optional): The ChromaDB collection to query. If None, uses default "my_documents" collection.
        
    Returns:
        dict: Query results containing documents, metadatas, and distances
    """
    try:
        if collection is None:
            # Initialize ChromaDB client and get default collection
            client = chromadb.Client(Settings(
                persist_directory="./chroma_db",
                is_persistent=True
            ))
            collection = client.get_collection("my_documents")
        
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results
    except Exception as e:
        print(f"Error querying vector store: {str(e)}")
        raise

# Example usage:
'''if __name__ == "__main__":
    # Extract text from PDF
    pdf_text = extract_text_from_pdf('MAY-22/Demo_Laptop_FAQ_Answers.pdf')

    # Create one chunk per line
    chunks = chunk_text(pdf_text)
    print(f"Created {len(chunks)} chunks from PDF")

    # Store in ChromaDB
    collection = create_vector_store(chunks, collection_name="my_documents")
    print("Documents stored in ChromaDB")

    # Query the collection (using default collection)
    results = query_vector_store("model", n_results=3)
    print("\nQuery Results:")
    print(f"Number of chunks retrieved: {len(results['documents'][0])}")
    for i, doc in enumerate(results['documents'][0]):
        print(f"\nChunk {i+1}:")
        print(doc)
'''
