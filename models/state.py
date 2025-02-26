from typing import TypedDict, List, Optional

class State(TypedDict):
    """
    State for the QA feedback system graph.
    """
    # Question and response
    question: str
    llm_response: str
    
    # Feedback data
    human_feedback: str
    is_validated: bool
    feedback_notes: str
    
    # Response history
    previous_responses: List[str]
    
    # Vector database info
    from_database: bool
    original_question: str
    is_identical: bool
    adapted_response: str

def create_initial_state(question: str) -> State:
    """
    Create an initial state for the given question.
    """
    return {
        "question": question,
        "llm_response": "",
        "human_feedback": "",
        "is_validated": False,
        "previous_responses": [],
        "feedback_notes": "",
        "from_database": False,
        "adapted_response": "",
        "original_question": "",
        "is_identical": False
    }