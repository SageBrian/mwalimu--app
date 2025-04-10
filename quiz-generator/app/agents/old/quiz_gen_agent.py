# --- Import Libraries ---
import asyncio
import os
from typing import Optional, List, Dict, Any, Literal
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
from app.models.pydantic_models import QuizState, WelcomeResponse, QuizQuestion, QuizResponse

#Prompts
from app.prompts.quiz_system_prompt import quiz_system_prompt
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
# Create structured output models using the LLM instance
# Ensure the schema passed matches the Pydantic model

quiz_model = llm.with_structured_output(QuizResponse)

async def generate_quiz_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate quiz questions for the given topic."""
    print("--- Running Generate Quiz Node ---")
    
    # Convert state dict to QuizState if it's not already
    current_state = state if isinstance(state, QuizState) else QuizState(**state)
    current_step = "generate"

    if not current_state.topic:
        print("Error: Generate node called without a topic.")
        return {
            "current_step": "error",
            "message_to_user": "I need a topic to generate a quiz. Could you please provide one?"
        }
    
    try:
        messages = [
            SystemMessage(content=quiz_system_prompt),
            HumanMessage(content=f"Generate a quiz about {current_state.topic}")
        ]
        
        print("Sending messages to LLM...")
        response = await quiz_model.ainvoke(messages)
        print(f"Raw LLM Response: {response}")
        
        # Handle both dict and QuizResponse object responses
        if isinstance(response, dict):
            response = QuizResponse(**response)
        
        # Format questions for display
        questions_text = "\n\n".join([
            f"Question {i+1}: {q.question}\n" +
            "\n".join(q.options) + "\n" +
            f"Correct Answer: {q.correct_answer}\n" +
            f"Explanation: {q.explanation}"
            for i, q in enumerate(response.questions)
        ])

        #empty the current_step
        current_step = ""
        
        # Prepare return state
        return_state = {
            "questions": response.questions,
            "current_step": "generate",
            "message_to_user": f"Here's your quiz about {current_state.topic}!\n\n{questions_text}"
        }
        print(f"Return State: {return_state}")
        return return_state
        
    except Exception as e:
        error_msg = f"Error generating quiz for topic '{current_state.topic}': {str(e)}"
        print(f"Debug - {error_msg}")
        return {
            "current_step": "error",
            "message_to_user": "I encountered an error generating the quiz. Could you please try again?",
            "questions": []
        }