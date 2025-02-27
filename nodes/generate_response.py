"""
Node responsible for generating initial responses to questions.
"""
from graph.state import State
from services.vector_db import VectorDBService
from services.llm_service import LLMService

class GenerateResponseNode:
    def __init__(self):
        self.vector_db_service = VectorDBService()
        self.llm_service = LLMService()
    
    def execute(self, state: State) -> State:
        """
        Executes the response generation node, which:
        1. Checks if there are similar validated responses in the database
        2. If found, adapts the existing response to the new question
        3. If not found, generates a new response using the LLM
        """
        print("Verifying similar questions")
        
        question = state["question"]
        
        # Search for similar responses
        similar_docs = self.vector_db_service.search_similar_responses(question)
        
        if similar_docs:
            doc, score = similar_docs[0]
            original_question = doc.metadata.get('question', 'similar question')
            validated_response = doc.page_content
            
            # Remove additional notes if present
            if "\n\nAdditional observations:" in validated_response:
                validated_response = validated_response.split("\n\nAdditional observations:")[0]
            
            # Check if the question is exactly identical
            if original_question.strip().lower() == question.strip().lower():
                print(f"Identical question found: '{original_question}'")
                return {
                    **state,
                    "llm_response": validated_response,
                    "original_question": original_question,
                    "from_database": True,
                    "is_identical": True
                }
            else:
                # Automatically adapt the response for non-identical questions
                print(f"Similar question found: '{original_question}'. Adapting...")
                adapted_response = self.llm_service.adapt_response(question, original_question, validated_response)
                
                return {
                    **state,
                    "llm_response": adapted_response,
                    "original_question": original_question,
                    "from_database": True,
                    "is_identical": False,
                    "adapted_response": adapted_response
                }
        
        # If no similar responses found, generate a new one        
        print("No answers found, generating new")
        
        llm_response = self.llm_service.generate_response(question)
        
        return {
            **state,
            "llm_response": llm_response,
            "previous_responses": state.get("previous_responses", []) + [llm_response],
            "from_database": False,
            "is_identical": False
        }

# Helper function to facilitate integration with the graph
def generate_llm_response(state: State) -> State:
    node = GenerateResponseNode()
    return node.execute(state)