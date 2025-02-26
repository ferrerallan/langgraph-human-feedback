import openai
from typing import Optional

from config import OPENAI_API_KEY, LLM_MODEL

class LLMService:
    LIGHT_BLUE = '\033[94m'
    END_COLOR = '\033[0m'

    def __init__(self):
        openai.api_key = OPENAI_API_KEY

    def blue_print(self, text):
        """Print text in light blue color."""
        print(f"{self.LIGHT_BLUE}{text}{self.END_COLOR}")
    
    def generate_response(self, question: str) -> str:
        """
        Generate a response to the given question using the LLM.
        
        Args:
            question: The question to answer
            
        Returns:
            Generated response
        """
        prompt = f"""
        Current question: {question}
        
        Please provide a detailed and accurate answer to the question above.
        """
        
        response = openai.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert assistant that provides accurate and helpful answers."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
    
    def regenerate_with_feedback(self, question: str, feedback: str) -> str:
        """
        Regenerate a response based on feedback.
        
        Args:
            question: Original question
            feedback: User feedback for improvement
            
        Returns:
            Regenerated response
        """
        prompt = f"""
        Question: {question}
        
        USER FEEDBACK: "{feedback}"
        
        Generate a new response that takes this feedback into account.
        Be very specific in following EXACTLY what the feedback asks.
        If the feedback mentions the response should be shorter, make it significantly shorter.
        If the feedback mentions limiting to certain aspects, focus ONLY on those aspects.
        """
        
        response = openai.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an assistant that rigorously follows user feedback. Adapt your response exactly as requested, without adding unrequested content."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content