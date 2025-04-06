import logging
from langgraph.graph import StateGraph, END

from app.graph.state import AppState
from app.agents.welcome_agent import run_welcome_agent # Import the new agent
# Import constants if needed (e.g., agent names)
from app.agents.welcome_agent import NEXT_AGENT_QUIZ_GENERATION

logger = logging.getLogger(__name__)

# --- Node Definitions ---
WELCOME_NODE = "run_welcome_agent"
QUIZ_GENERATION_NODE = "quiz_generation_node" # Placeholder for the next agent
# CLARIFICATION_NODE = "clarification_node"
# ERROR_HANDLER_NODE = "error_handler_node"

# --- Conditional Edge Logic based on Handoff ---

def route_from_welcome(state: AppState) -> str:
    """
    Routes flow based on the 'handoff' field within the 'last_agent_output'
    set by the WelcomeAgent.
    """
    logger.debug(f"--- Routing decision based on state after WelcomeAgent ---")
    last_output = state.get('last_agent_output')

    # Check for critical errors first (set directly in state)
    if state.get("error_message"):
        logger.warning(f"Error flag set in state: {state.get('error_message')}. Routing to END (or Error Node).")
        # return ERROR_HANDLER_NODE # Route to error handler when implemented
        return END # MVP: End on error

    if not last_output or not isinstance(last_output, dict):
         logger.error("Routing Error: last_agent_output is missing or not a dict. Routing to END.")
         return END

    # Check status within the last agent's output
    agent_status = last_output.get('status')
    if agent_status == 'error':
        logger.warning(f"Error status from {last_output.get('agent_name')}: {last_output.get('error_details')}. Routing to END.")
        return END # Or error node

    # Extract the handoff object (which should be a dict if serialized from Pydantic)
    handoff_data = last_output.get('handoff')

    if handoff_data and isinstance(handoff_data, dict):
        logger.info(f"Handoff detected from {last_output.get('agent_name')}.")
        # Extract the target agent name from the nested parameters
        target_agent_params = handoff_data.get('parameters')
        if target_agent_params and isinstance(target_agent_params, dict):
            target_agent_name = target_agent_params.get('agent_name')
            logger.info(f"Target agent specified in handoff: '{target_agent_name}'")

            if target_agent_name == NEXT_AGENT_QUIZ_GENERATION:
                logger.info(f"Routing to: {QUIZ_GENERATION_NODE} (Not Implemented Yet -> END)")
                # return QUIZ_GENERATION_NODE # <<< UNCOMMENT THIS when Quiz Node exists
                return END # MVP: End here for now
            # Add elif conditions for other target agents here...
            # elif target_agent_name == "quiz_interaction_agent":
            #     return QUIZ_INTERACTION_NODE
            else:
                logger.warning(f"Unknown target agent '{target_agent_name}' in handoff. Routing to END.")
                return END
        else:
            logger.warning("Handoff detected, but 'parameters' field is missing or invalid. Routing to END.")
            return END
    else:
        # No handoff specified - likely intent was unclear or task completed by WelcomeAgent (unlikely)
        logger.info(f"No handoff specified by {last_output.get('agent_name')} (Intent likely unclear or task finished). Routing to END.")
        # Could route to a CLARIFICATION_NODE if intent was unclear
        # if last_output.get('detected_intent') == INTENT_UNCLEAR:
        #     return CLARIFICATION_NODE
        return END


# --- Graph Builder Function ---
def build_graph() -> StateGraph:
    """
    Builds the LangGraph StateGraph definition using handoff-based routing.
    Checkpointer is applied during compilation.
    """
    workflow = StateGraph(AppState)

    # Add Nodes
    workflow.add_node(WELCOME_NODE, run_welcome_agent)
    # --- Add other agent nodes here later ---
    # workflow.add_node(QUIZ_GENERATION_NODE, quiz_generation_runnable)
    # workflow.add_node(CLARIFICATION_NODE, clarification_runnable)
    # workflow.add_node(ERROR_HANDLER_NODE, error_handler_runnable)

    # Set Entry Point
    workflow.set_entry_point(WELCOME_NODE)

    # Add Edges from Welcome Node using the new routing function
    workflow.add_conditional_edges(
        WELCOME_NODE,
        route_from_welcome, # Use the handoff-based routing function
        {
            # Map return values of route_from_welcome to target node names
            QUIZ_GENERATION_NODE: END,
            # CLARIFICATION_NODE: CLARIFICATION_NODE,
            # ERROR_HANDLER_NODE: ERROR_HANDLER_NODE,
            END: END # Route to END if function returns END or unknown path
        }
    )
    # Add edges from other nodes later...

    logger.info("LangGraph workflow definition built with handoff-based routing.")
    return workflow