from models.state import State
from services.vector_db_service import VectorDBService
from services.llm_service import LLMService
from services.adaptation_service import AdaptationService

from typing import Dict, Any

# Initialize services
vector_db_service = VectorDBService()
llm_service = LLMService()
adaptation_service = AdaptationService()

def generate_llm_response(state: State) -> State:
    """Graph node: Generate a response using LLM or retrieve from database."""
    print("Checking for similar validated responses")
    
    question = state["question"]
    
    # Search for similar responses
    similar_docs = vector_db_service.search_similar_responses(question)
    
    if similar_docs:
        # Found similar response(s)
        doc, score = similar_docs[0]
        
        # Process the response (adapt or use directly)
        response, is_identical = adaptation_service.process_similar_response(question, doc)
        original_question = doc.metadata.get('question', 'similar question')
        
        return {
            **state,
            "llm_response": response,
            "original_question": original_question,
            "from_database": True,
            "is_identical": is_identical
        }
    
    print("No similar validated response found, generating new response")
    
    # Generate a new response with the LLM
    llm_response = llm_service.generate_response(question)
    
    return {
        **state,
        "llm_response": llm_response,
        "previous_responses": state.get("previous_responses", []) + [llm_response],
        "from_database": False,
        "is_identical": False
    }

def get_human_feedback(state: State) -> State:
    """Graph node: Waiting point for human feedback."""
    print("Waiting for human feedback")
    return state

def evaluate_feedback(state: State) -> Dict[str, Any]:
    """Graph node: Evaluate feedback and decide next action."""
    print("Evaluating feedback")
    
    if state["is_validated"]:
        return {"next": "save_validated_response"}
    else:
        if len(state.get("previous_responses", [])) >= 3:
            return {"next": "END"}
        else:
            return {"next": "regenerate_response"}

def regenerate_response(state: State) -> State:
    """Graph node: Regenerate response based on feedback."""
    print("Regenerating response based on feedback")
    
    question = state["question"]
    feedback = state["feedback_notes"]
    previous_responses = state.get("previous_responses", [])
    
    # Generate new response with feedback
    new_response = llm_service.regenerate_with_feedback(question, feedback)
    
    return {
        **state,
        "llm_response": new_response,
        "previous_responses": previous_responses + [new_response],
        "from_database": False
    }

def save_validated_response(state: State) -> State:
    """Graph node: Save validated response to database."""
    print("Saving validated response in vector database")
    
    question = state["question"]
    validated_response = state["llm_response"]
    feedback_notes = state.get("feedback_notes", "")
    adapted_from = state.get("original_question", "") if state.get("from_database", False) else ""
    
    # Store the response
    vector_db_service.store_response(
        question=question,
        response=validated_response,
        feedback_notes=feedback_notes,
        adapted_from=adapted_from
    )
    
    return state