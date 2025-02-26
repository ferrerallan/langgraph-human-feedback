import openai
from typing import Tuple

from config import OPENAI_API_KEY, LLM_MODEL

class AdaptationService:
    GREEN = '\033[92m'
    END_COLOR = '\033[0m'

    def __init__(self):
        openai.api_key = OPENAI_API_KEY

    def green_print(self, text):
        """Print text in green color."""
        print(f"{self.GREEN}{text}{self.END_COLOR}")
    
    def adapt_response(self, question: str, stored_question: str, stored_response: str) -> str:
        """
        Adapt a stored response to a new question.
        
        Args:
            question: The new question
            stored_question: The original question for the stored response
            stored_response: The stored response to adapt
            
        Returns:
            Adapted response
        """
        self.green_print("Adapting stored response to current question")
        
        prompt = f"""
        I have a stored response for this question:
        "{stored_question}"
        
        The stored response is:
        "{stored_response}"
        
        Now I need to answer this new question:
        "{question}"
        
        Please adapt the stored response to answer the new question.
        Keep the same format and level of detail, but modify the content 
        to match the specific requirements of the new question.
        """
        
        response = openai.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert assistant that adapts existing answers to new contexts."},
                {"role": "user", "content": prompt}
            ]
        )
        
        adapted_response = response.choices[0].message.content
        self.green_print("Response adapted successfully to the new context.")
        
        return adapted_response
    
    def is_question_identical(self, question1: str, question2: str) -> bool:
        """
        Check if two questions are identical (ignoring case and whitespace).
        
        Args:
            question1: First question
            question2: Second question
            
        Returns:
            True if questions are identical
        """
        return question1.strip().lower() == question2.strip().lower()
    
    def process_similar_response(self, question: str, similar_doc) -> Tuple[str, bool]:
        """
        Process a similar response, either using it directly if identical
        or adapting it if similar.
        
        Args:
            question: Current question
            similar_doc: Similar document found in the database
            
        Returns:
            Tuple of (processed_response, is_identical)
        """
        original_question = similar_doc.metadata.get('question', 'similar question')
        validated_response = similar_doc.page_content
        
        # Remove additional notes if present
        if "\n\nAdditional notes:" in validated_response:
            validated_response = validated_response.split("\n\nAdditional notes:")[0]
        
        # Check if questions are identical
        if self.is_question_identical(original_question, question):
            self.green_print(f"Found identical question: '{original_question}'")
            return validated_response, True
        else:
            # Adapt the response for non-identical questions
            self.green_print(f"Found similar question: '{original_question}'. Adapting response...")
            adapted_response = self.adapt_response(question, original_question, validated_response)
            return adapted_response, False