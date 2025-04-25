
# mwalimubot/backedn/app/agents/router_agent.py

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
from app.models.pydantic_models import MwalimuBotState, Handoff, TutorParameters, RespondToUserParameters
#Prompts
from app.prompts.routing_agent_prompt import get_routing_agent_prompt

# Load environment variables
load_dotenv()

#logger
logger = logging.getLogger(__name__)


# Add the router_node function for direct import
async def router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Receives user_message and routes to the best Next agent."""
    # --- Print State Entering Node ---
    print("\n=== State Entering Router Node ===")
    print(state)
    print("=================================\n")
    # --- End Print ---

    print("=== Router Node Start Execution ===")

    # Convert state dict to QuizState if it's not already
    current_state = state if isinstance(state, MwalimuBotState) else MwalimuBotState(**state)

    # Increment router attempts
    current_state.router_attempts += 1
    print(f"Attempts: {current_state.router_attempts}")

    # Get user input from the prompt
    user_input = current_state.user_input
    print(f"User Input: {user_input}")

    try:
        # Get the system prompt with user input and conversation history
        system_prompt = get_routing_agent_prompt(
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
            #model="gpt-4o-mini-2024-07-18",
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
            "node_name": "router",
            "response": response
        })

        # Log state after router node (before returning changes)
        logger.info(f"State before processing response: {current_state}")

        # # Process handoff agents - extract from the response
        extracted_handoff_agents = []
        extracted_quiz_params = None
        message_to_student = None



        if response.handoff_agents:
            extracted_handoff_agents = response.handoff_agents
            print(f"Extracted handoff agents: {extracted_handoff_agents}")

            # Find quiz parameters if question_generator agent exists
            for agent in extracted_handoff_agents:
                if agent.agent_name == 'tutor_agent':
                    if isinstance(agent.agent_specific_parameters, TutorParameters):
                        extracted_tutor_params = agent.agent_specific_parameters
                        print(f"Extracted titor parameters: {extracted_tutor_params}")
                        break
                    else:
                        print(f"Warning: tutor_agent agent found but parameters are not TutorParameters type: {type(agent.agent_specific_parameters)}")
                if agent.agent_name == 'respond_to_user':
                    extracted_respond_params = agent.agent_specific_parameters
                    message_to_student = extracted_respond_params.message_to_student
                    print(f"Extracted respond parameters: {message_to_student}")
                    break

        print("=== Router Node End Execution (Success) ===")

        # Prepare the return dictionary
        return_state = {
            # "message_to_user": message_to_user,
            "node_history": current_state.node_history,
            "handoff_agents": [agent.agent_name for agent in extracted_handoff_agents],
            "handoff_agents_params": [agent.model_dump() for agent in extracted_handoff_agents],
            "current_step": "router",
            "router_attempts": current_state.router_attempts,
            "error_message": None,
            "conversation_history": current_state.conversation_history,
            "message_to_student": message_to_student if message_to_student else None
        }

        # --- Print State Exiting Node (Success) ---
        print("\n=== State Exiting Router Node (Success) ===")
        print(return_state)
        print("=========================================\n")
        # --- End Print ---

        return return_state
    
    except Exception as e:
        error_msg = f"Error in router node: {str(e)}"
        print(f"Debug - {error_msg}")

        # Update node_history with error message
        current_state.node_history.append({
            "node_name": "router",
            "response": f"Error: {error_msg}"
        })

        logger.info(f"State after error in router node: {current_state}")
        print("=== Router Node End Execution (Error) ===")

        # Prepare the return dictionary for error case
        return_state = {
            "node_history": current_state.node_history,
            "current_step": "error",
            "error_message": error_msg,
            "handoff_agents": [],
            "handoff_agents_params": [],
            "conversation_history": current_state.conversation_history
        }

        # --- Print State Exiting Node (Error) ---
        print("\n=== State Exiting router Node (Error) ===")
        print(return_state)
        print("=======================================\n")
        # --- End Print ---

        return return_state
