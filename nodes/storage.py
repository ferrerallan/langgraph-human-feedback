"""
Node responsible for storing validated responses in the database.
"""
from graph.state import State
from services.vector_db import VectorDBService

class StoreValidatedResponseNode:
    def __init__(self):
        self.vector_db_service = VectorDBService()
    
    def execute(self, state: State) -> State:
        """
        Saves the validated response in the vector database.
        """
        print("Storing on vector database")
        
        question = state["question"]
        validated_response = state["llm_response"]
        feedback_notes = state.get("feedback_notes", "")
        from_database = state.get("from_database", False)
        original_question = state.get("original_question", "")
        
        self.vector_db_service.add_validated_response(
            question=question,
            response=validated_response,
            feedback_notes=feedback_notes,
            original_question=original_question,
            from_database=from_database
        )
        
        return state

# Helper function to facilitate integration with the graph
def save_validated_response(state: State) -> State:
    node = StoreValidatedResponseNode()
    return node.execute(state)