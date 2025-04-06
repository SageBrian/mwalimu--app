from pydantic_ai import Agent
from pydantic import BaseModel, Field
from typing import Optional, List
import asyncio
import uuid

import os
from dotenv import load_dotenv
from pydantic_ai import  Agent
load_dotenv()

#from app.models.pydantic_models import KaribuAgentInput, KaribuAgentOutput
from app.shared_services.llm import call_llm_api
class KaribuAgentInput(BaseModel):
    conversation_id: Optional[str] = None
    user_response: Optional[str] = None

class KaribuAgentOutput(BaseModel):
    message_to_user: str
    fact: str = Field(description="A random fact about Kenya")

class ChakulaAgentInput(BaseModel):
    conversation_id: Optional[str] = None
    user_response: Optional[str] = None

class ChakulaAgentOutput(BaseModel):
    message_to_user: str
    fact: str = Field(description="A random fact about Tanzania")

karibu_agent = Agent(
    model= "openai:gpt-4o-mini",
    deps_type=KaribuAgentInput,
    result_type=KaribuAgentOutput,
    system_prompt="""
    You are a helpful AI assistant that can answer questions and help with tasks.
    """
)

chakula_agent = Agent(
    model= "openai:gpt-4o-mini",
    deps_type=ChakulaAgentInput,
    result_type=ChakulaAgentOutput,
    system_prompt="""
    You are a helpful AI assistant that can answer questions and help with tasks.
    Your answer should be in Swahili.
    """
)
async def run_karibu_agent(input_data: dict) -> KaribuAgentOutput:
    """Run the Karibu agent with the given input"""
    try:
        # Convert dict to dependencies model
        deps = KaribuAgentInput(**input_data)
        result = await karibu_agent.run(
            deps.user_response,
            deps=deps
        )
        print(result)
        return result.data
    except Exception as e:
        raise RuntimeError(f"Error running Karibu agent: {e}") from e

async def run_chakula_agent(input_data: dict) -> ChakulaAgentOutput:
    """Run the Chakula agent with the given input"""
    try:
        deps = ChakulaAgentInput(**input_data)
        result = await chakula_agent.run(
            deps.user_response,
            deps=deps   
        )
        print(result)
        return result.data
    except Exception as e:
        raise RuntimeError(f"Error running Chakula agent: {e}") from e

#test

#Langgraph

from langgraph.graph import StateGraph, START, END

class KaribuState(BaseModel):
    user_response: Optional[str] = None
    conversation_id: Optional[str] = None
    messages: List[dict] = Field(default_factory=list)
    node_history: List[dict] = Field(default_factory=list)
    agent_trace: List[str] = Field(default_factory=list)
    current_status: str = "started"
    karibu_agent_output: Optional[KaribuAgentOutput] = None
    chakula_agent_output: Optional[ChakulaAgentOutput] = None
    error_message: Optional[str] = None


def run_karibu_agent_sync(state):
    """Synchronous wrapper for the karibu agent"""
    try:
        input_data = {
            "conversation_id": state.conversation_id,
            "user_response": state.user_response
        }
        result = asyncio.run(run_karibu_agent(input_data))
        return result
    except Exception as e:
        raise RuntimeError(f"Error running Karibu agent: {e}") from e

def run_chakula_agent_sync(state):
    """Synchronous wrapper for the chakula agent"""
    try:
        input_data = {
            "conversation_id": state.conversation_id,
            "user_response": state.user_response
        }
        result = asyncio.run(run_chakula_agent(input_data))
        return result
    except Exception as e:
        raise RuntimeError(f"Error running Chakula agent: {e}") from e

# Modify graph definition to use sync functions
karibu_graph = StateGraph(KaribuState)

# Add nodes with sync functions
karibu_graph.add_node("karibu_agent", run_karibu_agent_sync)
karibu_graph.add_node("chakula_agent", run_chakula_agent_sync)

# Add edges
karibu_graph.add_edge(START, "karibu_agent")
karibu_graph.add_edge("karibu_agent", "chakula_agent")
karibu_graph.add_edge("chakula_agent", END)

# Compile graph
karibu_graph = karibu_graph.compile()

if __name__ == "__main__":
    user_response = input("Enter your response: ")
    
    # Create initial state
    initial_state = KaribuState(
        user_response=user_response,
        conversation_id=str(uuid.uuid4()),
        messages=[],
        node_history=[],
        agent_trace=[],
        current_status="started"
    )
    
    # Run the graph with initial state - remove asyncio.run()
    result = karibu_graph.invoke(initial_state)
    print("Final state:", result)






