import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END

from graph.nodes import (
    generate_llm_response,
    get_human_feedback,
    evaluate_feedback,
    regenerate_response,
    save_validated_response
)
from config import CHECKPOINT_DB

def build_graph():
    """
    Build and compile the feedback graph.
    
    Returns:
        Compiled graph
    """
    from models.state import State
    
    # Create a new graph
    builder = StateGraph(State)
    
    # Add nodes
    builder.add_node("generate_llm_response", generate_llm_response)
    builder.add_node("get_human_feedback", get_human_feedback)
    builder.add_node("evaluate_feedback", evaluate_feedback)
    builder.add_node("regenerate_response", regenerate_response)
    builder.add_node("save_validated_response", save_validated_response)
    
    # Add edges
    builder.add_edge(START, "generate_llm_response")
    builder.add_edge("generate_llm_response", "get_human_feedback")
    builder.add_edge("get_human_feedback", "evaluate_feedback")
    
    # Add conditional edges
    builder.add_conditional_edges(
        "evaluate_feedback",
        lambda x: x["next"],
        {
            "save_validated_response": "save_validated_response",
            "regenerate_response": "regenerate_response",
            END: END
        }
    )
    
    builder.add_edge("regenerate_response", "get_human_feedback")
    builder.add_edge("save_validated_response", END)
    
    # Set up checkpointing
    conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    memory = SqliteSaver(conn)
    
    # Compile the graph
    graph = builder.compile(checkpointer=memory, interrupt_before=["get_human_feedback"])
    
    # Generate graph visualization
    try:
        graph.get_graph().draw_mermaid_png(output_file_path="qa_feedback_graph.png")
        print("Graph visualization saved as 'qa_feedback_graph.png'")
    except Exception as e:
        print(f"Could not generate graph visualization: {e}")
    
    return graph