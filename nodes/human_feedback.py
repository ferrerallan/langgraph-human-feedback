"""
Node responsible for collecting human feedback about the response.
"""
from graph.state import State

class HumanFeedbackNode:
    def execute(self, state: State) -> State:
        """
        This node is an interruption point in the graph.
        It waits for human feedback, which will be provided
        externally and updated in the state.
        """
        print("Waiting for human feedback")
        return state

# Helper function to facilitate integration with the graph
def get_human_feedback(state: State) -> State:
    node = HumanFeedbackNode()
    return node.execute(state)