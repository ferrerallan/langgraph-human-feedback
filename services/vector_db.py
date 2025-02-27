"""
Service for handling the vector database.
"""
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import uuid
import types
from config import OPENAI_API_KEY, VECTOR_DB_PATH, SIMILARITY_THRESHOLD, MAX_SIMILAR_RESULTS

class VectorDBService:
    def __init__(self):
        self.db = self._initialize_vector_db()
    
    def _initialize_vector_db(self):
        """Initializes the vector database or creates a mock if there's an error."""
        try:
            embeddings = OpenAIEmbeddings()
            vector_db = Chroma(
                persist_directory=VECTOR_DB_PATH,
                embedding_function=embeddings
            )
            
            # Adds compatibility method if necessary
            if not hasattr(vector_db, 'similarity_search_with_score'):
                def similarity_search_with_score_compat(self, query, k=4, filter=None):
                    docs = self.similarity_search(query, k=k, filter=filter)
                    return [(doc, 0.5) for doc in docs]
                
                vector_db.similarity_search_with_score = types.MethodType(
                    similarity_search_with_score_compat, vector_db
                )
                
            return vector_db
            
        except Exception as e:
            # print(f"Warning: Error initializing vector database: {e}")
            # print("The system will continue to function, but responses will not be persisted.")
            
            return self._create_mock_db()
    
    def _create_mock_db(self):
        """Creates a mock of the vector database for when there are initialization errors."""
        class MockVectorDB:
            def add_texts(self, texts, metadatas=None):
                print("Simulating text storage in the database.")
                return ["mock_id"]
            
            def similarity_search(self, query, k=2, filter=None):
                print("Simulating similarity search in the database.")
                return []
            
            def similarity_search_with_score(self, query, k=2, filter=None):
                print("Simulating similarity search with score in the database.")
                return []
        
        return MockVectorDB()
    
    def search_similar_responses(self, question, k=MAX_SIMILAR_RESULTS, 
                                similarity_threshold=SIMILARITY_THRESHOLD):        
        try:
            results = self.db.similarity_search_with_score(
                query=question,
                k=k,
                filter={"validated": True}
            )
            
            relevant_results = []
            for doc, score in results:
                print(f"Question: '{doc.metadata.get('question', 'unknown')}' - Score: {score}")
                
                if score <= similarity_threshold:
                    relevant_results.append((doc, score))
            
            return relevant_results
        except Exception as e:
            print(f"Warning: Error searching for similar responses: {e}")
            return []
    
    def add_validated_response(self, question, response, feedback_notes="", original_question="", from_database=False):
        """Adds a validated response to the database."""
        final_document = response
        if feedback_notes:
            final_document += f"\n\nAdditional observations: {feedback_notes}"
        
        try:
            self.db.add_texts(
                texts=[final_document],
                metadatas=[{
                    "question": question,
                    "validated": True,
                    "id": str(uuid.uuid4()),
                    "adapted_from": original_question if from_database else ""
                }]
            )
            
            # Tries to persist the database
            if hasattr(self.db, 'persist'):
                self.db.persist()
            else:
                pass
                # print("Warning: The 'persist' method is not available in this version of Chroma.")
                
            # print(f"Validated response stored for the question: {question}")
            return True
        except Exception as e:
            print(f"Error saving response to the database: {e}")
            print("The response was processed, but may not have been persisted.")
            return False