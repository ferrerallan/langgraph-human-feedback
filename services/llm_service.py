"""
Service for interactions with language models.
"""
import openai
from config import LLM_MODEL

class LLMService:
    def __init__(self):
        self.model = LLM_MODEL
    
    def generate_response(self, question):
        """Generates a response to the question using the LLM."""
        prompt = f"""
        Current question: {question}
        
        Please provide a detailed and accurate answer to the question above.
        """
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert assistant that provides accurate and helpful answers."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
    
    def adapt_response(self, question, stored_question, stored_response):
        """Adapts a stored response to a new similar question."""
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
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert assistant that adapts existing answers to new contexts."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
    
    def regenerate_with_feedback(self, question, feedback, previous_responses=None):
        """Regenerates a response based on user feedback."""
        if previous_responses is None:
            previous_responses = []
            
        prompt = f"""
        Question: {question}

        USER FEEDBACK: "{feedback}"

        Generate a new response that takes this feedback into account.
        Be very specific in following EXACTLY what the feedback asks.
        If the feedback mentions the response should be shorter, make it significantly shorter.
        If the feedback mentions limiting to certain aspects, focus ONLY on those aspects.
        """

        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an assistant that rigorously follows user feedback. Adapt your response exactly as requested, without adding unrequested content."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content