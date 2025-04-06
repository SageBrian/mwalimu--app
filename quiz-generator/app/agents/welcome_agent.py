import logging
import os
import asyncio
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent

# Import necessary Pydantic models
from app.models.pydantic_models import (
    SingleHandoff,
    QuizGenerationHandoffParameters,
    QuizGenerationAgentInput
)
from app.graph.state import AppState

# --- Load environment variables ---
load_dotenv()

# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
AGENT_NAME = "WelcomeAgent"
INTENT_GENERATE_QUIZ = "request_quiz_generation"
INTENT_UNCLEAR = "unclear_or_general_query"
NEXT_AGENT_QUIZ_GENERATION = "quiz_generation_agent"

class WelcomeAgentDependencies(BaseModel):
    conversation_id: Optional[str] = None
    input_explanation: str

class WelcomeAgentResult(BaseModel):
    message_to_user: str = Field(description="Message to display to the user")
    detected_intent: str = Field(description="Detected intent from user input")
    handoff: Optional[SingleHandoff] = Field(None, description="Handoff parameters if needed")
    status: str = Field(default="success", description="Status of the agent's execution")
    error_details: Optional[str] = None

welcome_agent = Agent(
    'openai:gpt-4',
    deps_type=WelcomeAgentDependencies,
    result_type=WelcomeAgentResult,
    system_prompt="""
    You are a helpful AI assistant that welcomes users to a quiz generation system.
    Analyze the user's input to determine if they want to generate a quiz.
    
    If the user wants to generate a quiz:
    1. Set detected_intent to "request_quiz_generation"
    2. Create a friendly message acknowledging their request
    3. Include handoff parameters for the quiz generation agent
    
    If the user's intent is unclear or unrelated to quiz generation:
    1. Set detected_intent to "unclear_or_general_query"
    2. Create a helpful message asking for clarification
    3. Set handoff to None
    
    Always maintain a friendly and helpful tone in your message_to_user.
    """
)

async def run_welcome_agent(input_data: dict) -> WelcomeAgentResult:
    """Run the welcome agent with the given input"""
    try:
        # Convert dict to dependencies model
        deps = WelcomeAgentDependencies(**input_data)
        
        # Run agent with the input explanation as primary input
        result = await welcome_agent.run(
            deps.input_explanation,  # Primary input must be a string
            deps=deps  # Dependencies passed separately
        )
        return result.data
    except Exception as e:
        logger.error(f"Error running welcome agent: {str(e)}", exc_info=True)
        return WelcomeAgentResult(
            message_to_user="I apologize, but I encountered an error while processing your request. Please try again.",
            detected_intent=INTENT_UNCLEAR,
            status="error",
            error_details=str(e)
        )

def welcome_agent_runnable(state: AppState) -> AppState:
    """Wrapper for langgraph compatibility"""
    logger.info(f"--- Executing {AGENT_NAME} for conversation_id: {state.get('conversation_id')} ---")
    updated_state = state.copy()
    
    # Initialize state fields if they don't exist
    current_trace = updated_state.setdefault('agent_trace', [])
    messages_history = updated_state.setdefault('messages', [])
    node_history = updated_state.setdefault('node_history', [])
    
    input_text = updated_state.get('input_explanation')
    
    # Basic input validation
    if not input_text or not isinstance(input_text, str) or len(input_text.strip()) < 5:
        logger.warning(f"{AGENT_NAME}: Invalid or missing input explanation.")
        error_result = WelcomeAgentResult(
            message_to_user="I need a bit more information to help you. Could you please provide more details?",
            detected_intent=INTENT_UNCLEAR,
            status="error",
            error_details="Input explanation is missing or too short."
        )
        updated_state['current_status'] = "error"
        updated_state['error_message'] = error_result.error_details
        updated_state['last_agent_output'] = error_result.model_dump()
        node_history.append(error_result.model_dump())
        return updated_state
    
    try:
        # Run the welcome agent
        result = asyncio.run(run_welcome_agent(
            input_data={
                "conversation_id": state.get('conversation_id', ''),
                "input_explanation": input_text
            }
        ))
        
        # Update message history
        if not messages_history or messages_history[-1].get('content') != input_text:
            messages_history.append({"role": "user", "content": input_text})
        if result.message_to_user:
            messages_history.append({"role": "assistant", "content": result.message_to_user})
        
        # Update state
        updated_state['current_status'] = "processed_by_welcome"
        updated_state['error_message'] = None
        updated_state['last_agent_output'] = result.model_dump()
        node_history.append(result.model_dump())
        current_trace.append(f"{AGENT_NAME}: Processed successfully")
        
    except Exception as e:
        logger.error(f"{AGENT_NAME} execution failed: {str(e)}", exc_info=True)
        error_result = WelcomeAgentResult(
            message_to_user="I apologize, but something went wrong. Please try again.",
            detected_intent=INTENT_UNCLEAR,
            status="error",
            error_details=str(e)
        )
        updated_state['current_status'] = "error"
        updated_state['error_message'] = str(e)
        updated_state['last_agent_output'] = error_result.model_dump()
        node_history.append(error_result.model_dump())
        current_trace.append(f"{AGENT_NAME}: Failed execution")
    
    # Ensure all lists are updated in state
    updated_state['messages'] = messages_history
    updated_state['node_history'] = node_history
    updated_state['agent_trace'] = current_trace
    
    logger.info(f"--- {AGENT_NAME} Execution Complete ---")
    return updated_state

# Make it available for langgraph
#welcome_node = define_node("welcome", welcome_agent_runnable)