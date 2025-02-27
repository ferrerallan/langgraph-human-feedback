"""
Service for flow graph visualization.
"""

class VisualizationService:
    @staticmethod
    def generate_graph_image(graph, output_path="qa_feedback_graph.png"):
        """Generates an image of the graph using mermaid."""
        try:
            graph.get_graph().draw_mermaid_png(output_file_path=output_path)
            print(f"Visualization saved as '{output_path}'")
            return True
        except Exception as e:
            print(f"Error on generating visualization: {e}")
            return False