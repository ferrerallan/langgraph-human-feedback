import uuid
from typing import List, Tuple, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import types

from config import OPENAI_API_KEY, VECTOR_DB_DIR, SIMILARITY_THRESHOLD

class VectorDBService:
    def __init__(self):
        self.vector_db = self._initialize_db()
    
    YELLOW = '\033[93m'
    END_COLOR = '\033[0m'

    def yellow_print(self, text):
        """Print text in yellow color."""
        print(f"{self.YELLOW}{text}{self.END_COLOR}")
    
    def _initialize_db(self):
        """Initialize the vector database."""
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            vector_db = Chroma(
                persist_directory=VECTOR_DB_DIR,
                embedding_function=embeddings
            )
            
            # Add compatibility for similarity_search_with_score if not available
            if not hasattr(vector_db, 'similarity_search_with_score'):
                def similarity_search_with_score_compat(self, query, k=4, filter=None):
                    docs = self.similarity_search(query, k=k, filter=filter)
                    return [(doc, 0.5) for doc in docs]
                
                vector_db.similarity_search_with_score = types.MethodType(
                    similarity_search_with_score_compat, vector_db
                )
            
            self.yellow_print("Vector database initialized successfully.")
            return vector_db
            
        except Exception as e:
            print(f"Warning: Error initializing vector database: {e}")
            print("The system will continue working, but responses won't be persisted.")
            return self._create_mock_db()
    
    def _create_mock_db(self):
        """Create a mock database for when the real one fails to initialize."""
        class MockVectorDB:
            def add_texts(self, texts, metadatas=None):
                self.yellow_print("Simulating text storage in database.")
                return ["mock_id"]
            
            def similarity_search(self, query, k=2, filter=None):
                self.yellow_print("Simulating similarity search in database.")
                return []
            
            def similarity_search_with_score(self, query, k=2, filter=None):
                self.yellow_print("Simulating similarity search with score in database.")
                return []
        
        return MockVectorDB()
    
    def search_similar_responses(self, question: str, k: int = 2, 
                                 similarity_threshold: float = SIMILARITY_THRESHOLD) -> List[Tuple[Any, float]]:
        """
        Search for similar validated responses in the vector database.
        
        Args:
            question: The query question
            k: Number of results to return
            similarity_threshold: Minimum similarity score to consider
            
        Returns:
            List of tuples containing (document, score)
        """
        try:
            results = self.vector_db.similarity_search_with_score(
                query=question,
                k=k,
                filter={"validated": True}
            )
            
            relevant_results = []
            for doc, score in results:
                similarity = score
                
                self.yellow_print(f"Question: '{doc.metadata.get('question', 'unknown')}' - Score: {score}")
                
                if similarity <= similarity_threshold:
                    relevant_results.append((doc, score))
            
            return relevant_results
        except Exception as e:
            self.yellow_print(f"Warning: Error searching for similar responses: {e}")
            return []
    
    def store_response(self, question: str, response: str, 
                      feedback_notes: str = "", 
                      adapted_from: str = "") -> bool:
        """
        Store a validated response in the vector database.
        
        Args:
            question: The question being answered
            response: The validated response
            feedback_notes: Any additional notes from user feedback
            adapted_from: Original question if this is an adaptation
            
        Returns:
            Success flag
        """
        try:
            final_document = response
            if feedback_notes:
                final_document += f"\n\nAdditional notes: {feedback_notes}"
            
            metadata = {
                "question": question,
                "validated": True,
                "id": str(uuid.uuid4())
            }
            
            if adapted_from:
                metadata["adapted_from"] = adapted_from
            
            self.vector_db.add_texts(
                texts=[final_document],
                metadatas=[metadata]
            )
            
            if hasattr(self.vector_db, 'persist'):
                self.vector_db.persist()
            else:
                print("Warning: The 'persist' method is not available in this version of Chroma.")
                
            self.yellow_print(f"Validated response stored for the question: {question}")
            return True
            
        except Exception as e:
            print(f"Error saving response to database: {e}")
            print("The response was processed, but might not have been persisted.")
            return False