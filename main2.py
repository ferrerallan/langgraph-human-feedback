import sqlite3
import os
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, START, END
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import uuid
import json

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

try:
    embeddings = OpenAIEmbeddings()
    vector_db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    if not hasattr(vector_db, 'similarity_search_with_score'):
        def similarity_search_with_score_compat(self, query, k=4, filter=None):
            docs = self.similarity_search(query, k=k, filter=filter)
            return [(doc, 0.5) for doc in docs]
        
        import types
        vector_db.similarity_search_with_score = types.MethodType(
            similarity_search_with_score_compat, vector_db
        )    
    
except Exception as e:
    print(f"Warning: Error initializing vector database: {e}")
    print("The system will continue working, but responses won't be persisted.")
    
    class MockVectorDB:
        def add_texts(self, texts, metadatas=None):
            print("Simulating text storage in database.")
            return ["mock_id"]
        
        def similarity_search(self, query, k=2, filter=None):
            print("Simulating similarity search in database.")
            return []
        
        def similarity_search_with_score(self, query, k=2, filter=None):
            print("Simulating similarity search with score in database.")
            return []
    
    vector_db = MockVectorDB()

class State(TypedDict):
    question: str
    llm_response: str
    human_feedback: str
    is_validated: bool
    previous_responses: list
    feedback_notes: str
    from_database: bool
    adapted_response: str

def search_similar_validated_responses(question, k=2, similarity_threshold=0.25):
    try:
        results = vector_db.similarity_search_with_score(
            query=question,
            k=k,
            filter={"validated": True}
        )
        
        relevant_results = []
        for doc, score in results:
            similarity = score
            
            print(f"Question: '{doc.metadata.get('question', 'unknown')}' - Score: {score}")
            
            if similarity >= similarity_threshold:
                relevant_results.append((doc, score))
        
        return relevant_results
    except Exception as e:
        print(f"Warning: Error searching for similar responses: {e}")
        return []

def adapt_database_response(question, stored_question, stored_response):
    print("Adapting stored response to current question")
    
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
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert assistant that adapts existing answers to new contexts."},
            {"role": "user", "content": prompt}
        ]
    )
    
    adapted_response = response.choices[0].message.content
    print("Response adapted successfully to the new context.")
    
    return adapted_response

def generate_llm_response(state: State) -> State:
    print("Checking for similar validated responses")
    
    question = state["question"]
    
    similar_docs = search_similar_validated_responses(question, similarity_threshold=0.25)
    
    if similar_docs:
        doc, score = similar_docs[0]
        original_question = doc.metadata.get('question', 'similar question')
        validated_response = doc.page_content
        
        if "\n\nAdditional notes:" in validated_response:
            validated_response = validated_response.split("\n\nAdditional notes:")[0]
        
        # Check if the question is exactly identical
        if original_question.strip().lower() == question.strip().lower():
            print(f"Found identical question: '{original_question}'")
            return {
                **state,
                "llm_response": validated_response,
                "original_question": original_question,
                "from_database": True,
                "is_identical": True
            }
        else:
            # Automatically adapt the response for non-identical questions
            print(f"Found similar question: '{original_question}'. Adapting response...")
            adapted_response = adapt_database_response(question, original_question, validated_response)
            
            return {
                **state,
                "llm_response": adapted_response,
                "original_question": original_question,
                "from_database": True,
                "is_identical": False,
                "adapted_response": adapted_response
            }
    
    print("No similar validated response found, generating new response")
    
    prompt = f"""
    Current question: {question}
    
    Please provide a detailed and accurate answer to the question above.
    """
    
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert assistant that provides accurate and helpful answers."},
            {"role": "user", "content": prompt}
        ]
    )
    
    llm_response = response.choices[0].message.content
    
    return {
        **state,
        "llm_response": llm_response,
        "previous_responses": state.get("previous_responses", []) + [llm_response],
        "from_database": False,
        "is_identical": False
    }

def get_human_feedback(state: State) -> State:
    print("Waiting for human feedback")
    return state

def evaluate_feedback(state: State) -> dict:
    print("Evaluating feedback")
    
    if state["is_validated"]:
        return {"next": "save_validated_response"}
    else:
        if len(state.get("previous_responses", [])) >= 3:
            return {"next": END}
        else:
            return {"next": "regenerate_response"}

def regenerate_response(state: State) -> State:
    print("Regenerating response based on feedback")
    
    question = state["question"]
    feedback = state["feedback_notes"]
    previous_responses = state.get("previous_responses", [])
    
    prompt = f"""
    Question: {question}
    
    USER FEEDBACK: "{feedback}"
    
    Generate a new response that takes this feedback into account.
    Be very specific in following EXACTLY what the feedback asks.
    If the feedback mentions the response should be shorter, make it significantly shorter.
    If the feedback mentions limiting to certain aspects, focus ONLY on those aspects.
    """
    
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an assistant that rigorously follows user feedback. Adapt your response exactly as requested, without adding unrequested content."},
            {"role": "user", "content": prompt}
        ]
    )
    
    new_response = response.choices[0].message.content
    
    return {
        **state,
        "llm_response": new_response,
        "previous_responses": previous_responses + [new_response],
        "from_database": False
    }

def save_validated_response(state: State) -> State:
    print("Saving validated response in vector database")
    
    question = state["question"]
    validated_response = state["llm_response"]
    feedback_notes = state.get("feedback_notes", "")
    
    final_document = validated_response
    if feedback_notes:
        final_document += f"\n\nAdditional notes: {feedback_notes}"
    
    try:
        vector_db.add_texts(
            texts=[final_document],
            metadatas=[{
                "question": question,
                "validated": True,
                "id": str(uuid.uuid4()),
                "adapted_from": state.get("original_question", "") if state.get("from_database", False) else ""
            }]
        )
        
        if hasattr(vector_db, 'persist'):
            vector_db.persist()
        else:
            print("Warning: The 'persist' method is not available in this version of Chroma.")
            
        print(f"Validated response stored for the question: {question}")
    except Exception as e:
        print(f"Error saving response to database: {e}")
        print("The response was processed, but might not have been persisted.")
    
    return state

builder = StateGraph(State)

builder.add_node("generate_llm_response", generate_llm_response)
builder.add_node("get_human_feedback", get_human_feedback)
builder.add_node("evaluate_feedback", evaluate_feedback)
builder.add_node("regenerate_response", regenerate_response)
builder.add_node("save_validated_response", save_validated_response)

builder.add_edge(START, "generate_llm_response")
builder.add_edge("generate_llm_response", "get_human_feedback")
builder.add_edge("get_human_feedback", "evaluate_feedback")

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

conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)

graph = builder.compile(checkpointer=memory, interrupt_before=["get_human_feedback"])

try:
    graph.get_graph().draw_mermaid_png(output_file_path="qa_feedback_graph.png")
    print("Graph visualization saved as 'qa_feedback_graph.png'")
except Exception as e:
    print(f"Could not generate graph visualization: {e}")

def run_qa_feedback_system(question):
    if not question.strip():
        print("\nThe question cannot be empty. Please try again.")
        return
    
    thread_id = str(uuid.uuid4())
    thread = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
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
    
    print(f"\nNew question: {question}\n")
    print("Searching for an answer... Please wait.")
    
    try:
        events = list(graph.stream(initial_state, thread, stream_mode="values"))
        
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
        
        if from_database:
            if is_identical:
                print(f"\nFound identical question in database.")
                print("-" * 50)
                print(llm_response)
                print("-" * 50)
                print("\nThis is an exact match for your question!")
            else:
                print(f"\nAdapted response based on similar question:")
                print(f"Original question: {original_question}")
                print("-" * 50)
                print(llm_response)
                print("-" * 50)
                print("\nThis response was adapted from a similar question.")
        else:
            print("\nResponse generated by LLM:")
            print("-" * 50)
            print(llm_response)
            print("-" * 50)
        
        # Ask for feedback regardless of the source
        print("\n" + "=" * 50)
        print("FEEDBACK NEEDED")
        print("=" * 50)
        print("Please evaluate the response above:")
        
        valid_input = None
        while valid_input is None:
            user_input = input("\nIs the response valid? (yes/no): ").lower().strip()
            if user_input in ["yes", "y"]:
                valid_input = True
            elif user_input in ["no", "n"]:
                valid_input = False
            else:
                print("Please answer with 'yes' or 'no'.")
        
        if valid_input:
            feedback_notes = input("Would you like to add any observation to the response? (optional): ")
            
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
            next_events = list(graph.stream(None, thread, stream_mode="values"))
            
            print("\n" + "=" * 50)
            print("SUCCESS: Response validated and stored!")
            print("=" * 50)
            print("This response has been saved in the vector database")
            print("and will be used as a reference for similar")
            print("questions in the future.")
            print("=" * 50)
            
            return
        
        # Handle negative feedback and regeneration
        attempt_count = 1  # Already showed the first response
        current_response = llm_response
        is_validated = False
        
        while attempt_count < 3 and not is_validated:
            attempt_count += 1
            
            feedback_notes = input("Please explain what needs to be improved: ")
            if not feedback_notes.strip():
                feedback_notes = "Response does not meet expectations. Please generate a new response."
            
            print(f"\nGenerating new response based on your feedback...")
            
            graph.update_state(
                thread, 
                {
                    "human_feedback": "rejected",
                    "is_validated": False,
                    "feedback_notes": feedback_notes
                }, 
                as_node="get_human_feedback"
            )
            
            next_events = list(graph.stream(None, thread, stream_mode="values"))
            
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
            
            print(f"\nRegenerated response (attempt {attempt_count}):")
            print("-" * 50)
            print(current_response)
            print("-" * 50)
            
            print("\n" + "=" * 50)
            print("FEEDBACK NEEDED")
            print("=" * 50)
            print("Please evaluate the response above:")
            
            valid_input = None
            while valid_input is None:
                user_input = input("\nIs the response valid? (yes/no): ").lower().strip()
                if user_input in ["yes", "y"]:
                    valid_input = True
                elif user_input in ["no", "n"]:
                    valid_input = False
                else:
                    print("Please answer with 'yes' or 'no'.")
            
            if valid_input:
                is_validated = True
                feedback_notes = input("Would you like to add any observation to the response? (optional): ")
                
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
                next_events = list(graph.stream(None, thread, stream_mode="values"))
                
                print("\n" + "=" * 50)
                print("SUCCESS: Response validated and stored!")
                print("=" * 50)
                print("This response has been saved in the vector database")
                print("and will be used as a reference for similar")
                print("questions in the future.")
                print("=" * 50)
                
                break
            
            if attempt_count >= 3:
                print("\n" + "=" * 50)
                print("WARNING: Attempt limit reached")
                print("=" * 50)
                print("After 3 attempts without validation, the flow will end.")
                print("=" * 50)
                break
        
        if not is_validated:
            print("\n" + "=" * 50)
            print("ENDED: Flow completed without validation")
            print("=" * 50)
            print("After several attempts, we couldn't reach")
            print("a response that could be validated.")
            print("=" * 50)
    
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Please try again with a different question.")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RESPONSE SYSTEM WITH HUMAN VALIDATION")
    print("=" * 60)
    print("This system allows you to ask questions and validate the answers.")
    print("PROCESS FLOW:")
    print("1. You ask a question")
    print("2. The system searches for similar questions or generates a new response")
    print("3. For similar questions, the system automatically adapts the response")
    print("   unless the question is exactly identical")
    print("4. You validate the response or suggest improvements")
    print("5. If validated - Response stored for future use")
    print("   If rejected - System generates a new response with your feedback")
    print("=" * 60)
    
    while True:
        question = input("\nEnter your question (or 'exit' to end): ").strip()
        if question.lower() in ["exit", "quit"]:
            print("\nEnding the system. Thank you for using it!")
            break
            
        run_qa_feedback_system(question)