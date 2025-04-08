"""
Graph Builder Module
"""
from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END, START
from langsmith import trace
from langchain_core.messages import SystemMessage, HumanMessage

from app.models.pydantic_models import QuizState, WelcomeResponse, QuizResponse

#Nodes
from app.agents.welcome_agent import welcome_node
from app.agents.quiz_gen_agent import generate_quiz_node
from app.agents.respond_to_user import respond_to_user_node

def should_respond_to_user(state: Dict[str, Any]) -> bool:
    """Determine if we should continue to the respond node."""
    # Convert state dict to QuizState if it's not already
    current_state = state if isinstance(state, QuizState) else QuizState(**state)
    
    # If we have a message to show to user, route to respond
    if current_state.message_to_user:
        return True
    
    # Default to end if no other conditions are met
    return False

def should_continue_to_generate(state: Dict[str, Any]) -> bool:
    """Determine if we should continue to the generate node."""
    # Convert state dict to QuizState if it's not already
    current_state = state if isinstance(state, QuizState) else QuizState(**state)
    
    # If we have a topic and user input, proceed to generate
    if current_state.topic and current_state.user_input:
        return True
    
    return False

def continue_after_respond(state: Dict[str, Any]) -> str:
    """Determine next step after responding to user"""
    current_state = state if isinstance(state, QuizState) else QuizState(**state)
    
    # If we have a topic and user wants to generate, move to generate
    if current_state.topic and current_state.user_input and "generate" in current_state.user_input.lower():
        return "generate"
    
    # If we have a topic but no generate request, stay in welcome
    if current_state.topic:
        return "welcome"
    
    # If we're in welcome step, stay in welcome
    if current_state.current_step == "welcome":
        return "welcome"
    
    # If we're in error state, end
    if current_state.current_step == "error":
        return "error"
    
    # Default to welcome
    return "welcome"

def should_continue_to_welcome(state: Dict[str, Any]) -> str:
    """Determine if we should continue to welcome or end"""
    current_state = state if isinstance(state, QuizState) else QuizState(**state)
    
    # If we have a message to show to user, route to respond
    if current_state.message_to_user:
        return "respond"
    
    # If we have too many attempts, end
    if current_state.welcome_attempts >= 3:
        return "end"
    
    # If we have a topic, move to topic validation
    if current_state.topic:
        return "topic_validation"
    
    # If we're in welcome step, stay in welcome
    if current_state.current_step == "welcome":
        return "welcome"
    
    # Otherwise, continue to welcome
    return "welcome"

def build_graph() -> StateGraph:
    """Build the quiz generation workflow graph."""
    # Define the workflow
    workflow = StateGraph(QuizState)

    # Add nodes
    workflow.add_node("respond", respond_to_user_node)
    workflow.add_node("welcome", welcome_node)
    workflow.add_node("generate", generate_quiz_node)

    # Set entry point
    workflow.set_entry_point("respond")

    # Add edges
    workflow.add_edge("respond", "welcome")

    # Add conditional edges
    workflow.add_conditional_edges(
        "welcome",
        should_respond_to_user,
        {
            True: "respond",
            False: "generate"
        }
    )

    workflow.add_conditional_edges(
        "respond",
        continue_after_respond,
        {
            "welcome": "welcome",
            "generate": "generate",
            "error": END
        }
    )
    
    return workflow