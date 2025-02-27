"""
Helper functions for the QA system with feedback.
"""
import uuid
from typing import Dict, Any, List

def create_initial_state(question: str) -> Dict[str, Any]:
    """
    Creates the initial state for the workflow.
    
    Args:
        question: The user's question
        
    Returns:
        A dictionary with the initial state
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

def create_thread_config() -> Dict[str, Any]:
    """
    Creates a thread configuration for the graph.
    
    Returns:
        A dictionary with thread configuration
    """
    thread_id = str(uuid.uuid4())
    return {"configurable": {"thread_id": thread_id}}

def format_display_response(response: str, source_type: str, original_question: str = "") -> str:
    """
    Formats the response for display.
    
    Args:
        response: The response to be displayed
        source_type: The source type ("database", "adapted", "generated")
        original_question: The original question (only for adapted responses)
        
    Returns:
        The formatted response for display
    """
    header = "-" * 50 + "\n"
    footer = "\n" + "-" * 50
    
    if source_type == "database":
        return f"{header}{response}{footer}\n\nExact correspondence for your question!"
    elif source_type == "adapted":
        return f"Adapted answer:\n {response}"
    else:
        return f"{header}{response}{footer}"

def format_feedback_prompt() -> str:
    """
    Formats the prompt to request feedback.
    
    Returns:
        The formatted feedback prompt
    """
    return "\n" + "=" * 50 + "\nFEEDBACK NEEDED\n" + "=" * 50 + "\nPlease validate the answer:"

def format_success_message() -> str:
    """
    Formats the success message when validating a response.
    
    Returns:
        The formatted success message
    """
    return "\n" + "=" * 50 + "\nSUCCESS: Saved into the vector database!\n"

def format_warning_message() -> str:
    """
    Formats the warning message when the attempt limit is reached.
    
    Returns:
        The formatted warning message
    """
    return "\n" + "=" * 50 + "\nWARNING: Limit reached"

def format_end_message() -> str:
    """
    Formats the flow ending message without validation.
    
    Returns:
        The formatted ending message
    """
    return "\n" + "=" * 50 + "\nFINISHED: Completed flow without validation\n"

def get_yes_no_input(prompt: str) -> bool:
    """
    Gets a yes/no input from the user.
    
    Args:
        prompt: The message to be displayed
        
    Returns:
        True for yes, False for no
    """
    valid_input = None
    while valid_input is None:
        user_input = input(prompt).lower().strip()
        if user_input in ["sim", "s", "yes", "y"]:
            valid_input = True
        elif user_input in ["não", "nao", "n", "no"]:
            valid_input = False
        else:
            print("Answer with 'yes' or 'no'.")
    
    return valid_input

def print_welcome_message() -> None:
    """
    Prints the system welcome message.
    """
    print("\n" + "=" * 60)
    print("Human-in-the-Loop Learning System")
    print("=" * 60)