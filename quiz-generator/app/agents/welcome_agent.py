# --- Import Libraries ---
import asyncio
import os
from typing import Optional, List, Dict, Any, Literal, Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
# Using MemorySaver for simple in-memory state between turns in the script
from langgraph.checkpoint.memory import MemorySaver
from langsmith import Client
from langsmith import trace

# No need for these specific imports if not used in prompts
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# Instead, we'll format prompts directly for simplicity here
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

#Models
from app.models.pydantic_models import QuizState, WelcomeResponse, QuizQuestion, QuizResponse, HandoffParameters

#Prompts
from app.prompts.welcome_system_prompt import get_welcome_system_prompt

# Load environment variables
load_dotenv()

# Configure LangSmith (Keep as is)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "quiz-generator-refined" # Changed project name slightly

# Initialize LangSmith client (Keep as is)
langsmith_client = Client()

# Initialize LLM (Keep as is)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

#

    # Helper to clear interaction fields for the next turn
def clear_interaction(self):
    self.user_input = None
    self.response_to_user = None
    self.error_message = None
    return self


# Create structured output models using the LLM instance
# Ensure the schema passed matches the Pydantic model
welcome_model = llm.with_structured_output(HandoffParameters)

def handle_handoff_response(response: Dict[str, Any], current_state: QuizState) -> Dict[str, Any]:
    """Handle different types of handoff responses from the LLM."""
    if not isinstance(response, dict):
        return {
            "response_to_user": "I'm not sure how to help with that. Could you please rephrase?",
            "message_to_user": "I'm not sure how to help with that. Could you please rephrase?",
            "topic": None,
            "current_step": "welcome",
            "welcome_attempts": current_state.welcome_attempts,
            "error_message": None,
            "user_input": None
        }
    
    agent_name = response.get('agent_name')
    params = response.get('agent_specific_parameters', {})
    
    # Define handoff handlers
    handlers = {
        'respond_to_user': lambda: {
            "response_to_user": params.get('message_to_user', ''),
            "message_to_user": params.get('message_to_user', ''),
            "topic": None,
            "current_step": "welcome",
            "welcome_attempts": current_state.welcome_attempts,
            "error_message": None,
            "user_input": None
        },
        'question_generator': lambda: {
            "topic": params.get('topic'),
            "current_step": "generate",
            "welcome_attempts": current_state.welcome_attempts,
            "error_message": None,
            "user_input": None,
            "quiz_parameters": params
        }
    }
    
    # Get the appropriate handler or use default
    handler = handlers.get(agent_name)
    if handler:
        return handler()
    
    # Default response if no matching handler
    return {
        "response_to_user": "I'm not sure how to help with that. Could you please rephrase?",
        "message_to_user": "I'm not sure how to help with that. Could you please rephrase?",
        "topic": None,
        "current_step": "welcome",
        "welcome_attempts": current_state.welcome_attempts,
        "error_message": None,
        "user_input": None
    }

async def welcome_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Welcome the user and extract a topic."""
    print("=== Welcome Node Start ===")
    
    # Convert state dict to QuizState if it's not already
    current_state = state if isinstance(state, QuizState) else QuizState(**state)
    print(f"Current State: {current_state}")
    
    # Increment welcome attempts
    current_state.welcome_attempts += 1
    print(f"Attempts: {current_state.welcome_attempts}")
    
    # Get user input from the prompt
    user_input = current_state.user_input
    print(f"User Input: {user_input}")
    
    try:
        # Get the system prompt with user input and conversation history
        system_prompt = get_welcome_system_prompt(
            user_input=user_input,
            conversation_history=current_state.conversation_history
        )
        
        # Send message to LLM
        print("Sending messages to LLM...")
        response = await welcome_model.ainvoke(user_input)
        print(f"Raw LLM Response: {response}")
        
        # Handle the response using the handler function
        return_state = handle_handoff_response(response, current_state)
        print(f"Return State: {return_state}")
        return return_state
        
    except Exception as e:
        error_msg = f"Error in welcome node: {str(e)}"
        print(f"Debug - {error_msg}")
        return {
            "error_message": error_msg,
            "current_step": "error",
            "message_to_user": "I encountered an error. Could you please try again?",
            "user_input": None
        }


