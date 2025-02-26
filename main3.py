import traceback
from models.state import create_initial_state
from graph.builder import build_graph
from services.feedback_service import FeedbackService
from config import MAX_ATTEMPTS
from utils.logger import log_question, log_response, log_feedback, log_error

def run_qa_feedback_system(question):
    """
    Run the QA feedback system with the given question.
    
    Args:
        question: The question to process
    """
    # Initialize services
    feedback_service = FeedbackService()
    
    # Input validation
    if not question.strip():
        print("\nThe question cannot be empty. Please try again.")
        return
    
    # Create a thread ID for this conversation
    import uuid
    thread_id = str(uuid.uuid4())
    thread = {"configurable": {"thread_id": thread_id}}
    
    # Create initial state
    initial_state = create_initial_state(question)
    
    # Log the question
    log_question(question)
    
    print(f"\nNew question: {question}\n")
    print("Searching for an answer... Please wait.")
    
    try:
        # Build and get the graph
        graph = build_graph()
        
        # Run the graph until the first interruption
        events = list(graph.stream(initial_state, thread, stream_mode="values"))
        
        # Extract response information
        llm_response = None
        from_database = False
        original_question = ""
        is_identical = False
        
        for event in events:
            if isinstance(event, dict) and "llm_response" in event and event["llm_response"]:
                llm_response = event["llm_response"]
                from_database = event.get("from_database", False)
                original_question = event.get("original_question", "")
                is_identical = event.get("is_identical", False)
        
        if not llm_response:
            print("\nError: Could not generate a response.")
            return
        
        # Log the response
        source = "database" if from_database else "LLM"
        log_response(llm_response, source)
        
        # Display the response
        feedback_service.display_response(
            response=llm_response,
            from_database=from_database,
            is_identical=is_identical,
            original_question=original_question
        )
        
        # Get initial feedback
        is_validated = feedback_service.get_validation_feedback()
        log_feedback(is_validated)
        
        if is_validated:
            # Get additional notes if validated
            feedback_notes = feedback_service.get_additional_notes()
            
            # Update state with positive feedback
            graph.update_state(
                thread, 
                {
                    "human_feedback": "validated",
                    "is_validated": True,
                    "feedback_notes": feedback_notes
                }, 
                as_node="get_human_feedback"
            )
            
            print("\nStoring validated response in the database...")
            # Process the validated response
            graph.stream(None, thread, stream_mode="values")
            
            # Show success message
            feedback_service.show_validation_success()
            return
        
        # Handle negative feedback and regeneration cycle
        current_response = llm_response
        attempt_count = 1  # Already showed the first response
        
        while attempt_count < MAX_ATTEMPTS and not is_validated:
            attempt_count += 1
            
            # Get improvement feedback
            feedback_notes = feedback_service.get_improvement_feedback()
            log_feedback(False, feedback_notes)
            
            print(f"\nGenerating new response based on your feedback...")
            
            # Update state with negative feedback
            graph.update_state(
                thread, 
                {
                    "human_feedback": "rejected",
                    "is_validated": False,
                    "feedback_notes": feedback_notes
                }, 
                as_node="get_human_feedback"
            )
            
            # Continue the graph to generate a new response
            next_events = list(graph.stream(None, thread, stream_mode="values"))
            
            # Find the new response in events
            new_response = None
            for event in next_events:
                if isinstance(event, dict) and "llm_response" in event:
                    new_response = event.get("llm_response")
                    if new_response and new_response != current_response:
                        current_response = new_response
                        log_response(new_response, "regeneration")
                        break
            
            if not new_response:
                print("\nError generating new response.")
                break
            
            # Display the regenerated response
            feedback_service.display_response(
                response=current_response,
                from_database=False,
                attempt=attempt_count
            )
            
            # Get feedback on the regenerated response
            is_validated = feedback_service.get_validation_feedback()
            log_feedback(is_validated)
            
            if is_validated:
                # Validated after regeneration
                feedback_notes = feedback_service.get_additional_notes()
                
                # Update state with positive feedback
                graph.update_state(
                    thread, 
                    {
                        "human_feedback": "validated",
                        "is_validated": True,
                        "llm_response": current_response,
                        "feedback_notes": feedback_notes
                    }, 
                    as_node="get_human_feedback"
                )
                
                print("\nStoring validated response in the database...")
                graph.stream(None, thread, stream_mode="values")
                
                feedback_service.show_validation_success()
                break
            
            # Check if we've reached the attempt limit
            if attempt_count >= MAX_ATTEMPTS:
                feedback_service.show_attempt_limit_warning()
                break
        
        # If we exited the loop without validation
        if not is_validated:
            feedback_service.show_no_validation_message()
    
    except Exception as e:
        log_error(e)
        print(f"\nError during execution: {str(e)}")
        traceback.print_exc()
        print("Please try again with a different question.")

def main():
    """Main function to run the application."""
    print("\n" + "=" * 60)
    print("RESPONSE SYSTEM WITH HUMAN VALIDATION")
    print("=" * 60)
    
    while True:
        question = input("\nEnter your question (or 'exit' to end): ").strip()
        if question.lower() in ["exit", "quit"]:
            print("\nEnding the system. Thank you for using it!")
            break
            
        run_qa_feedback_system(question)

if __name__ == "__main__":
    main()