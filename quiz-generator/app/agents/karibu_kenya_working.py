import asyncio
import uuid
import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import json

# Assuming PydanticAI and necessary LLM libraries are installed
# pip install pydantic-ai langchain-openai openai python-dotenv langgraph
from pydantic_ai import Agent # Renamed to avoid confusion with "Agent" concept
from langchain_openai import ChatOpenAI # PydanticAI often uses langchain wrappers

# LangGraph imports
from langgraph.graph import StateGraph, START, END

# --- Load Environment Variables ---
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY not found in environment variables.")

# --- Pydantic Models (Agent I/O and State) ---
class KenyaAgentInput(BaseModel):
    conversation_id: Optional[str] = None
    user_response: Optional[str] = None

class KenyaAgentOutput(BaseModel):
    message_to_user: str
    fact: List[str] = Field(description="A random fact about Kenya",min_items=2,max_items=3)

class TanzaniaAgentInput(BaseModel):
    conversation_id: Optional[str] = None
    user_response: Optional[str] = None

class TanzaniaAgentOutput(BaseModel):
    message_to_user: str
    fact: List[str] = Field(description="A random fact about Tanzania",min_items=2,max_items=3)

class TriageAgentInput(BaseModel):
    conversation_id: Optional[str] = None
    user_response: Optional[str] = None
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    node_history: List[Dict[str, Optional[str]]] = Field(default_factory=list)
  
class TriageAgentOutput(BaseModel):
    message_to_user: Optional[str] = Field(description="Optional message to the user", default=None)
    agent_to_use: str = Field(description="The agent to use based on the user's response",choices=["kenya","tanzania"])

# --- PydanticAI Agent Definitions ---

# Use a compatible LLM wrapper (e.g., from langchain)
llm_engine = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

kenya_agent = Agent(
    model="gpt-4o-mini",
    deps_type=KenyaAgentInput,
    result_type=KenyaAgentOutput,
    system_prompt="""
    You are a helpful AI assistant. Your name is Kenya Agent. You provide interesting facts about Kenya.
    Respond warmly and include the fact.
    """
)

tanzania_agent = Agent(
    model="gpt-4o-mini",
    deps_type=TanzaniaAgentInput,
    result_type=TanzaniaAgentOutput,
    system_prompt="""
    You are a helpful AI assistant. Your name is Tanzania Agent. You provide interesting facts about Tanzania.
    Respond warmly in Swahili and include the fact.
    """
)

triage_agent = Agent(
    model="gpt-4o-mini",
    deps_type=TriageAgentInput,
    result_type=TriageAgentOutput,
    system_prompt="""
    You are a helpful AI assistant. Your name is Triage. You are responsible for triaging the user's response and determining which agent to use.
    if the user's response is about Kenya, return "kenya".
    if the user's response is about Tanzania, return "tanzania".

    """
)

# --- Async PydanticAI Agent Runner Functions ---
# These functions remain largely the same, just ensuring they use the initialized agents
async def run_kenya_agent(input_data: dict) -> KenyaAgentOutput:
    """Run the Kenya agent with the given input"""
    try:
        deps = KenyaAgentInput(**input_data)
        
        result = await kenya_agent.run( # Using await assumes PydanticAI's run or underlying call is awaitable
            deps.user_response,
            deps=deps
        )
        print(f"DEBUG: Kenya Agent Raw Result: {result}")
        # PydanticAI >= 1.0 returns a result object, access data via .result or .data depending on version
        # Adjust based on your PydanticAI version. Assuming older version returning the model directly.
        if isinstance(result, KenyaAgentOutput):
             return result
        elif hasattr(result, 'data') and isinstance(result.data, KenyaAgentOutput):
             return result.data
        elif hasattr(result, 'result') and isinstance(result.result, KenyaAgentOutput):
             return result.result
        else:
            raise TypeError(f"Unexpected result type from Kenya agent: {type(result)}")

    except Exception as e:
        print(f"Error details in run_kenya_agent: {e}")
        raise RuntimeError(f"Error running Kenya agent: {e}") from e

async def run_tanzania_agent(input_data: dict) -> TanzaniaAgentOutput:
    """Run the Tanzania agent with the given input"""
    try:
        deps = TanzaniaAgentInput(**input_data)
        result = await tanzania_agent.run( # Using await assumes PydanticAI's run or underlying call is awaitable
            deps.user_response,
            deps=deps
        )
        print(f"DEBUG: Tanzania Agent Raw Result: {result}")
        # Handle different PydanticAI result structures
        if isinstance(result, TanzaniaAgentOutput):
             return result
        elif hasattr(result, 'data') and isinstance(result.data, TanzaniaAgentOutput):
             return result.data
        elif hasattr(result, 'result') and isinstance(result.result, TanzaniaAgentOutput):
             return result.result
        else:
            raise TypeError(f"Unexpected result type from Tanzania agent: {type(result)}")

    except Exception as e:
        print(f"Error details in run_tanzania_agent: {e}")
        raise RuntimeError(f"Error running Tanzania agent: {e}") from e

async def run_triage_agent(input_data: dict) -> TriageAgentOutput:
    """Run the Triage agent with the given input"""
    try:
        deps = TriageAgentInput(**input_data)
        result = await triage_agent.run(
            deps.user_response,
            deps=deps
        )
        print(f"DEBUG: Triage Agent Raw Result: {result}")
        return result
    except Exception as e:
        print(f"Error details in run_triage_agent: {e}")
        raise RuntimeError(f"Error running Triage agent: {e}") from e
    
# --- LangGraph State Definition ---
class AppState(BaseModel):
    user_response: Optional[str] = None
    conversation_id: Optional[str] = None
    messages: List[Dict[str, str]] = Field(default_factory=list) # Use Dict for JSON compatibility
    node_history: List[Dict[str, Optional[str]]] = Field(default_factory=list) # Track node outputs/errors
    agent_trace: List[str] = Field(default_factory=list)
    current_status: str = "started"
    kenya_agent_output: Optional[KenyaAgentOutput] = None # Store the structured output
    tanzania_agent_output: Optional[TanzaniaAgentOutput] = None # Store the structured output
    triage_agent_output: Optional[TriageAgentOutput] = None # Store the structured output
    error_message: Optional[str] = None

    class Config:
        # Allow extra fields if necessary during runtime, though explicit updates are better
        extra = 'ignore'


# --- Refactored Async LangGraph Node Functions ---
# Nodes now MUST be async def and return a dictionary of state updates

async def triage_node(state: AppState) -> Dict:
    """Async LangGraph node for the Triage agent."""
    print("--- Entering Triage Node ---")
    node_name = "triage_agent"
    updated_trace = state.agent_trace + [node_name]
    updates: Dict[str, Any] = {"agent_trace": updated_trace}    

    try:
        input_data = {
            "conversation_id": state.conversation_id,
            "user_response": state.user_response
        }   
        result = await run_triage_agent(input_data)
        
        # Extract data from AgentRunResult
        triage_output = result.data if hasattr(result, 'data') else result
        
        updates.update({
            "triage_agent_output": triage_output,
            "messages": state.messages + [
                {"role": "user", "content": state.user_response or ""},
                {"role": "assistant", "content": triage_output.message_to_user} if triage_output.message_to_user else None
            ],
            "node_history": state.node_history + [{"node": node_name, "output": triage_output.model_dump()}],
            "current_status": f"processed_by_{triage_output.agent_to_use}",  # Use the agent_to_use field
            "error_message": None
        })
        print(f"--- Exiting Triage Node (Success) ---")

    except Exception as e:
        print(f"!!! Error in Triage Node: {e}")
        updates.update({
            "error_message": f"Error in {node_name}: {e}",
            "current_status": f"error_in_{node_name}",
            "node_history": state.node_history + [{"node": node_name, "error": str(e)}]
        })
        print(f"--- Exiting Triage Node (Error) ---")

    return updates


async def kenya_node(state: AppState) -> Dict: # Mark as async, specify return type hint as Dict
    """Async LangGraph node for the Kenya agent."""
    print("--- Entering Kenya Node ---")
    node_name = "kenya_agent"
    updated_trace = state.agent_trace + [node_name]
    updates: Dict[str, Any] = {"agent_trace": updated_trace} # Start building updates

    try:
        input_data = {
            "conversation_id": state.conversation_id,
            "user_response": state.user_response
        }
        # Await the async agent runner
        result: KenyaAgentOutput = await run_kenya_agent(input_data)

        # Prepare state updates correctly
        updates.update({
            "kenya_agent_output": result,
            "messages": state.messages + [
                {"role": "user", "content": state.user_response or ""}, # Handle potential None
                {"role": "assistant", "content": result.message_to_user}
            ],
            "node_history": state.node_history + [{"node": node_name, "output": result.model_dump_json()}],
            "current_status": "processed_by_kenya",
            "error_message": None # Clear any previous error
        })
        print(f"--- Exiting Kenya Node (Success) ---")

    except Exception as e:
        print(f"!!! Error in Kenya Node: {e}")
        # Update state with error information
        updates.update({
            "error_message": f"Error in {node_name}: {e}",
            "current_status": f"error_in_{node_name}",
            "node_history": state.node_history + [{"node": node_name, "error": str(e)}]
        })
        print(f"--- Exiting Kenya Node (Error) ---")

    return updates # Return the dictionary of updates

async def tanzania_node(state: AppState) -> Dict: # Mark as async, return Dict
    """Async LangGraph node for the Tanzania agent."""
    print("--- Entering Tanzania Node ---")
    node_name = "tanzania_agent"
    updated_trace = state.agent_trace + [node_name]
    updates: Dict[str, Any] = {"agent_trace": updated_trace}

    # Decide input for Tanzania: Use original user input for this simple linear flow
    user_input_for_tanzania = state.user_response

    if state.error_message:
        print("--- Skipping Tanzania Node due to previous error ---")
        # If there was an error in the previous step, maybe just pass through
        updates["current_status"] = "skipped_tanzania_due_to_error"
        return updates

    try:
        input_data = {
            "conversation_id": state.conversation_id,
            "user_response": user_input_for_tanzania
        }
        result: TanzaniaAgentOutput = await run_tanzania_agent(input_data)

        updates.update({
            "tanzania_agent_output": result,
            "messages": state.messages + [ # Append only Tanzania's response
                {"role": "assistant", "content": result.message_to_user}
            ],
            "node_history": state.node_history + [{"node": node_name, "output": result.model_dump_json()}],
            "current_status": "processed_by_tanzania",
            # Don't clear error message here, let it persist if set previously
        })
        print(f"--- Exiting Tanzania Node (Success) ---")

    except Exception as e:
        print(f"!!! Error in Tanzania Node: {e}")
        updates.update({
            "error_message": f"Error in {node_name}: {e}", # Overwrite or append? Overwrite for now.
            "current_status": f"error_in_{node_name}",
            "node_history": state.node_history + [{"node": node_name, "error": str(e)}]
        })
        print(f"--- Exiting Tanzania Node (Error) ---")

    return updates

async def triage_decision_node(state: AppState) -> str:
    """Async LangGraph node for the Triage decision node."""
    print("--- Entering Triage Decision Node ---")
    
    try:
        if state.triage_agent_output:
            agent_to_use = state.triage_agent_output.agent_to_use
            print(f"--- Exiting Triage Decision Node (Success) - Routing to: {agent_to_use} ---")
            
            # Return the string directly for routing
            if agent_to_use == "kenya":
                return "processed_by_kenya"
            elif agent_to_use == "tanzania":
                return "processed_by_tanzania"
            else:
                print(f"Invalid agent to use: {agent_to_use}")
                return "error_in_triage_decision"
        else:
            print("No triage agent output found")
            return "error_in_triage_decision"
            
    except Exception as e:
        print(f"!!! Error in Triage Decision Node: {e}")
        return "error_in_triage_decision"

# --- LangGraph Graph Definition ---
# Use the async node functions
workflow = StateGraph(AppState)

workflow.add_node("kenya_agent", kenya_node) # Use the async node function
workflow.add_node("tanzania_agent", tanzania_node) # Use the async node function
workflow.add_node("triage_agent", triage_node) # Use the async node function

# Define the graph flow (same linear flow)
workflow.add_edge(START, "triage_agent")
#add conditional edge   
workflow.add_conditional_edges(
    "triage_agent",
    triage_decision_node,
    {
        "processed_by_kenya": "kenya_agent",
        "processed_by_tanzania": "tanzania_agent"
    }
)
workflow.add_edge("kenya_agent", END)
workflow.add_edge("tanzania_agent", END)
# Compile the graph
# No checkpointer added in this refactor, as per instructions
compiled_graph = workflow.compile()

# --- Main Execution Block ---
async def main():

    """Asynchronous main function to run the graph."""
    #get user input repeatedly until the user enters "exit" or "quit"
    while True:
        user_response = input("Enter your response: ")
        if user_response.lower() in ["exit", "quit"]:
            break
        
        # Create initial state
        initial_state = AppState(
            user_response=user_response,
            conversation_id=str(uuid.uuid4()),
            current_status="started"
        )

        print("\n--- Invoking Graph ---")
        # Run the graph asynchronously using ainvoke
        final_state = await compiled_graph.ainvoke(initial_state)
        print("\n--- Graph Execution Complete ---")

        print("\nFinal State:")
        # Convert AddableValuesDict to a regular dict for printing
        print(json.dumps(dict(final_state), indent=2, default=str))

if __name__ == "__main__":
    # Run the main async function using asyncio.run() ONCE
    asyncio.run(main())