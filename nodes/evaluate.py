"""
Node responsible for evaluating human feedback and deciding the next action.
"""
from graph.state import State
from langgraph.graph import END

class EvaluateFeedbackNode:
    def execute(self, state: State) -> dict:
        """
        Evaluates human feedback and decides the next action:
        - If validated, saves the response
        - If rejected, regenerates the response (up to 3 attempts)
        - If rejected more than 3 times, ends the flow
        """
        print("Checking feedback")
        
        if state["is_validated"]:
            return {"next": "save_validated_response"}
        else:
            if len(state.get("previous_responses", [])) >= 3:
                return {"next": END}
            else:
                return {"next": "regenerate_response"}

# Helper function to facilitate integration with the graph
def evaluate_feedback(state: State) -> dict:
    node = EvaluateFeedbackNode()
    return node.execute(state)