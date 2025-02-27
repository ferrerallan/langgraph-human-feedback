"""
Main entry point for the QA system with feedback.
"""
import sys
import traceback
from graph.builder import GraphBuilder
from utils.helpers import (
    create_initial_state,
    create_thread_config,
    format_display_response,
    format_feedback_prompt,
    format_success_message,
    format_warning_message,
    format_end_message,
    get_yes_no_input,
    print_welcome_message
)

def run_qa_feedback_system(question):
    """
    Runs the QA system with feedback for a specific question.
    
    Args:
        question: The user's question
    """
    if not question.strip():
        print("\nQuestion shouldn't be empty.")
        return
    
    # Initialize the graph
    builder = GraphBuilder()
    graph = builder.build()
    
    # Create the initial state and thread configuration
    initial_state = create_initial_state(question)
    thread = create_thread_config()
    
    print(f"\nNew question: {question}\n")
    print("Searching for answer...")
    
    try:
        # Executes the initial flow until the interruption point (human_feedback)
        events = list(graph.stream(initial_state, thread, stream_mode="values"))
        
        llm_response = None
        from_database = False
        original_question = ""
        is_identical = False
        
        # Extract information from the most recent event
        for event in events:
            if isinstance(event, dict) and "llm_response" in event and event["llm_response"]:
                llm_response = event["llm_response"]
                from_database = event.get("from_database", False)
                original_question = event.get("original_question", "")
                is_identical = event.get("is_identical", False)
        
        if not llm_response:
            print("\nError: Could not generate a response.")
            return
        
        # Display the response with appropriate format
        if from_database:
            if is_identical:
                print(f"\nIdentical question found!")
                print(format_display_response(llm_response, "database"))
            else:
                print(format_display_response(llm_response, "adapted", original_question))
        else:
            print("\nAnswer by LLM:")
            print(format_display_response(llm_response, "generated"))
        
        # Request feedback from the user
        print(format_feedback_prompt())
        
        is_valid = get_yes_no_input("\nThe answer is valid? (yes/no): ")
        
        if is_valid:
            feedback_notes = ""
            
            # Update the state with positive feedback
            graph.update_state(
                thread, 
                {
                    "human_feedback": "validated",
                    "is_validated": True,
                    "feedback_notes": feedback_notes
                }, 
                as_node="get_human_feedback"
            )
            
            print("\nSaving answer...")
            next_events = list(graph.stream(None, thread, stream_mode="values"))
            
            print(format_success_message())
            
            return
        
        # Handle negative feedback and regeneration
        attempt_count = 1  # Already showed the first response
        current_response = llm_response
        is_validated = False
        
        while attempt_count < 3 and not is_validated:
            attempt_count += 1
            
            feedback_notes = input("Explain what can be improved: ")
            if not feedback_notes.strip():
                feedback_notes = "Answer doesn't meet expectations."
            
            print(f"\nGenerating new answer based on feedback...")
            
            # Update the state with negative feedback
            graph.update_state(
                thread, 
                {
                    "human_feedback": "rejected",
                    "is_validated": False,
                    "feedback_notes": feedback_notes
                }, 
                as_node="get_human_feedback"
            )
            
            # Continue the flow
            next_events = list(graph.stream(None, thread, stream_mode="values"))
            
            # Extract the new response
            new_response = None
            for event in next_events:
                if isinstance(event, dict) and "llm_response" in event:
                    new_response = event.get("llm_response")
                    if new_response and new_response != current_response:
                        current_response = new_response
                        break
            
            if not new_response:
                print("\nError generating new response.")
                break
            
            # Display the new response
            print(f"\nRegenerated answer (attempt {attempt_count}):")
            print(format_display_response(current_response, "generated"))
            
            # Request feedback for the new response
            print(format_feedback_prompt())
            
            is_valid = get_yes_no_input("\nThe answer is valid (yes/no): ")
            
            if is_valid:
                is_validated = True
                feedback_notes = ""
                
                # Update the state with positive feedback
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
                
                print("\nStoring in database...")
                next_events = list(graph.stream(None, thread, stream_mode="values"))
                
                print(format_success_message())
                
                break
            
            if attempt_count >= 3:
                print(format_warning_message())
                break
        
        if not is_validated:
            print(format_end_message())
    
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        traceback.print_exc()
        print("Please try again with a different question.")

def main():
    """Main function that starts the system."""
    print_welcome_message()
    
    while True:
        question = input("\nWrite your question or 'exit': ").strip()
        if question.lower() in ["sair", "exit", "quit"]:
            print("\nFinishing system...")
            break
            
        run_qa_feedback_system(question)

if __name__ == "__main__":
    main()