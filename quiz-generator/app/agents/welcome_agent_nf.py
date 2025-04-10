#-- Welcome Agent for Quiz Generator, without Frameworks

# --- Import Libraries ---
import asyncio
import os
import logging
from typing import Optional, List, Dict, Any, Literal, Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field

#Shared Services
from app.shared_services.llm import call_llm_api

#Models
from app.models.pydantic_models import QuizState, WelcomeResponse, QuizQuestion, HandoffParameters, Handoff, RespondToUserParameters, QuizGenParameters
#Prompts
from app.prompts.welcome_system_prompt import get_welcome_system_prompt

# Load environment variables
load_dotenv()

#logger
logger = logging.getLogger(__name__)


# Add the welcome_node function for direct import
async def welcome_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Welcome the user and extract a topic."""
    # --- Print State Entering Node ---
    print("\n=== State Entering Welcome Node ===")
    print(state)
    print("=================================\n")
    # --- End Print ---

    print("=== Welcome Node Start Execution ===")

    # Convert state dict to QuizState if it's not already
    current_state = state if isinstance(state, QuizState) else QuizState(**state)

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
        
        # Call LLM with the prompt & structured output
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input if user_input else ""}
        ]
        
        response = call_llm_api(
            messages=messages,
            model="gpt-4o-mini-2024-07-18",
            temperature=0.7,
            response_format=Handoff
        )

        print(f"Raw LLM Response: {response}")

        # Ensure response is a Handoff object
        if not isinstance(response, Handoff):
            print(f"Debug - Unexpected response type: {type(response)}")
            raise TypeError("LLM response was not the expected Handoff object.")

        # Update node_history with the parsed LLM response object
        current_state.node_history.append({
            "node_name": "welcome",
            "response": response
        })

        # Log state after welcome node (before returning changes)
        logger.info(f"State before processing response: {current_state}")

        # Process handoff agents - extract from the response
        extracted_handoff_agents = []
        extracted_quiz_params = None
        message_to_user = None

        if response.handoff_agents:
            extracted_handoff_agents = response.handoff_agents
            print(f"Extracted handoff agents: {extracted_handoff_agents}")

            # Find quiz parameters if question_generator agent exists
            for agent in extracted_handoff_agents:
                if agent.agent_name == 'question_generator':
                    if isinstance(agent.agent_specific_parameters, QuizGenParameters):
                        extracted_quiz_params = agent.agent_specific_parameters
                        print(f"Extracted quiz parameters: {extracted_quiz_params}")
                        break
                    else:
                        print(f"Warning: question_generator agent found but parameters are not QuizGenParameters type: {type(agent.agent_specific_parameters)}")
                if agent.agent_name == 'respond_to_user':
                    extracted_respond_params = agent.agent_specific_parameters
                    message_to_user = extracted_respond_params.message_to_user
                    print(f"Extracted respond parameters: {message_to_user}")
                    break

        print("=== Welcome Node End Execution (Success) ===")

        # Prepare the return dictionary
        return_state = {
            "message_to_user": message_to_user,
            "node_history": current_state.node_history,
            "handoff_agents": [agent.agent_name for agent in extracted_handoff_agents],
            "handoff_agents_params": [agent.model_dump() for agent in extracted_handoff_agents],
            "quiz_parameters": extracted_quiz_params.model_dump() if extracted_quiz_params else None,
            "current_step": "welcome",
            "welcome_attempts": current_state.welcome_attempts,
            "error_message": None,
            "user_input": None,
            "conversation_history": current_state.conversation_history
        }

        # --- Print State Exiting Node (Success) ---
        print("\n=== State Exiting Welcome Node (Success) ===")
        print(return_state)
        print("=========================================\n")
        # --- End Print ---

        return return_state
    
    except Exception as e:
        error_msg = f"Error in welcome node: {str(e)}"
        print(f"Debug - {error_msg}")

        # Update node_history with error message
        current_state.node_history.append({
            "node_name": "welcome",
            "response": f"Error: {error_msg}"
        })

        logger.info(f"State after error in welcome node: {current_state}")
        print("=== Welcome Node End Execution (Error) ===")

        # Prepare the return dictionary for error case
        return_state = {
            "node_history": current_state.node_history,
            "current_step": "error",
            "error_message": error_msg,
            "user_input": None,
            "handoff_agents": [],
            "quiz_parameters": None,
            "welcome_attempts": current_state.welcome_attempts,
            "conversation_history": current_state.conversation_history
        }

        # --- Print State Exiting Node (Error) ---
        print("\n=== State Exiting Welcome Node (Error) ===")
        print(return_state)
        print("=======================================\n")
        # --- End Print ---

        return return_state
