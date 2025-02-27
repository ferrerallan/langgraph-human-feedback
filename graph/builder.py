"""
Flow graph builder for the QA system with feedback.
"""
import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from graph.state import State
from nodes.generate_response import generate_llm_response
from nodes.human_feedback import get_human_feedback
from nodes.evaluate import evaluate_feedback
from nodes.regenerate import regenerate_response
from nodes.storage import save_validated_response
from config import SQLITE_DB_PATH
from services.visualization import VisualizationService

class GraphBuilder:
    def __init__(self):
        """Initializes the graph builder."""
        self.builder = StateGraph(State)
        self.conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
        self.memory = SqliteSaver(self.conn)
        self.visualization_service = VisualizationService()
    
    def build(self):
        """
        Builds the graph for the QA system with feedback.
        
        Main flow:
        1. Start -> generate_llm_response: Generates an initial response
        2. generate_llm_response -> get_human_feedback: Gets human feedback
        3. get_human_feedback -> evaluate_feedback: Evaluates feedback
        4a. evaluate_feedback -> save_validated_response: If validated
        4b. evaluate_feedback -> regenerate_response: If not validated (up to 3 attempts)
        4c. evaluate_feedback -> END: If rejected more than 3 times
        5a. regenerate_response -> get_human_feedback: Get feedback for the new response
        5b. save_validated_response -> END: Ends the flow after saving
        """
        # Adds the nodes
        self.builder.add_node("generate_llm_response", generate_llm_response)
        self.builder.add_node("get_human_feedback", get_human_feedback)
        self.builder.add_node("evaluate_feedback", evaluate_feedback)
        self.builder.add_node("regenerate_response", regenerate_response)
        self.builder.add_node("save_validated_response", save_validated_response)
        
        # Adds the edges
        self.builder.add_edge(START, "generate_llm_response")
        self.builder.add_edge("generate_llm_response", "get_human_feedback")
        self.builder.add_edge("get_human_feedback", "evaluate_feedback")
        
        # Adds conditional edges based on feedback evaluation
        self.builder.add_conditional_edges(
            "evaluate_feedback",
            lambda x: x["next"],
            {
                "save_validated_response": "save_validated_response",
                "regenerate_response": "regenerate_response",
                END: END
            }
        )
        
        # Adds the remaining edges
        self.builder.add_edge("regenerate_response", "get_human_feedback")
        self.builder.add_edge("save_validated_response", END)
        
        # Compiles the graph
        self.graph = self.builder.compile(
            checkpointer=self.memory, 
            interrupt_before=["get_human_feedback"]
        )
        
        # Generates graph visualization
        self.visualization_service.generate_graph_image(self.graph)
        
        return self.graph