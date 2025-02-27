"""
Node responsible for regenerating responses based on feedback.
"""
from graph.state import State
from services.llm_service import LLMService

class RegenerateResponseNode:
    def __init__(self):
        self.llm_service = LLMService()
    
    def execute(self, state: State) -> State:
        """
        Regenerates a response based on user feedback.
        """
        print("Regenerating based on feedback")
        
        question = state["question"]
        feedback = state["feedback_notes"]
        previous_responses = state.get("previous_responses", [])
        
        new_response = self.llm_service.regenerate_with_feedback(
            question, 
            feedback, 
            previous_responses
        )
        
        return {
            **state,
            "llm_response": new_response,
            "previous_responses": previous_responses + [new_response],
            "from_database": False
        }

# Helper function to facilitate integration with the graph
def regenerate_response(state: State) -> State:
    node = RegenerateResponseNode()
    return node.execute(state)