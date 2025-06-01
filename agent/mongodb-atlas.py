from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class MongoDBVectorStore:
    def __init__(self, connection_string: str, database_name: str, collection_name: str):
        """
        Initialize MongoDB connection and collection
        
        Args:
            connection_string (str): MongoDB Atlas connection string
            database_name (str): Name of the database
            collection_name (str): Name of the collection
        """
        self.client: MongoClient = MongoClient(connection_string)
        self.db: Database = self.client[database_name]
        self.collection: Collection = self.db[collection_name]
        
        # Create index for vector similarity search
        self.collection.create_index([("vector", "2dsphere")])
    
    def store_embedding(self, 
                       text: str, 
                       vector: List[float], 
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store text and its vector embedding in MongoDB
        
        Args:
            text (str): The text content
            vector (List[float]): The vector embedding
            metadata (Optional[Dict[str, Any]]): Additional metadata to store
            
        Returns:
            str: The ID of the inserted document
        """
        document = {
            "text": text,
            "vector": vector,
            "metadata": metadata or {}
        }
        
        result = self.collection.insert_one(document)
        return str(result.inserted_id)
    
    def find_similar_vectors(self, 
                           query_vector: List[float], 
                           limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar vectors using cosine similarity
        
        Args:
            query_vector (List[float]): The query vector to compare against
            limit (int): Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of similar documents with their similarity scores
        """
        # Convert query vector to numpy array for efficient computation
        query_vector = np.array(query_vector)
        
        # Get all vectors from the collection
        documents = list(self.collection.find({}, {"vector": 1, "text": 1, "metadata": 1}))
        
        # Calculate cosine similarity for each document
        similarities = []
        for doc in documents:
            doc_vector = np.array(doc["vector"])
            # Calculate cosine similarity
            similarity = np.dot(query_vector, doc_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
            )
            similarities.append((doc, similarity))
        
        # Sort by similarity score in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        return [
            {
                "id": str(doc["_id"]),
                "text": doc["text"],
                "metadata": doc.get("metadata", {}),
                "similarity_score": float(score)
            }
            for doc, score in similarities[:limit]
        ]
    
    def close(self):
        """Close the MongoDB connection"""
        self.client.close()

# Example usage:
if __name__ == "__main__":
    # Get MongoDB connection string from environment variable
    connection_string = os.getenv("MONGODB_URI")
    if not connection_string:
        raise ValueError("MONGODB_URI environment variable not set")
    
    # Initialize vector store
    vector_store = MongoDBVectorStore(
        connection_string=connection_string,
        database_name="vector_db",
        collection_name="embeddings"
    )
    
    # Example: Store an embedding
    text = "This is a sample text"
    vector = [0.1, 0.2, 0.3, 0.4]  # Example vector
    metadata = {"source": "example", "timestamp": "2024-03-20"}
    
    doc_id = vector_store.store_embedding(text, vector, metadata)
    print(f"Stored document with ID: {doc_id}")
    
    # Example: Find similar vectors
    query_vector = [0.1, 0.2, 0.3, 0.4]
    similar_docs = vector_store.find_similar_vectors(query_vector, limit=3)
    print("Similar documents:", similar_docs)
    
    # Close the connection
    vector_store.close()