"""
This is a quiz generator agent that generates a quiz based on the user's response.
It has these agents:
1 Welcome Agent - This is the first agent that greets the user and asks for the user's name & respond to chitchat & optionally handoff
 to a specialized agent based on the user's response.
2 Quiz Generator Agent - This agent generates a quiz based on the user's response
3 Answer Checker Agent - This agent checks the user's answer against the correct answer

"""

import asyncio
import uuid
import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import json


from pydantic_ai import Agent # Renamed to avoid confusion with "Agent" concept
from langchain_openai import ChatOpenAI # PydanticAI often uses langchain wrappers

# LangGraph imports
from langgraph.graph import StateGraph, START, END

# --- Load Environment Variables ---
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY not found in environment variables.")

# --- Pydantic Models (Agent I/O and State) ---
class WelcomeAgentInput(BaseModel):
    conversation_id: Optional[str] = None
    user_response: Optional[str] = None
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    node_history: List[Dict[str, Optional[str]]] = Field(default_factory=list)

class WelcomeAgentOutput(BaseModel):
    message_to_user: Optional[str] = Field(None, description="An optional message to the user, if there is an agent to use, you don't need to respond to the user")
    agent_to_use: Optional[str] = Field(None, description="The agent to use based on the user's response, sometimes you don't need to use an agent, in that case return None",choices=["quiz_generator_agent","answer_checker_agent"])

class QuizGeneratorAgentInput(BaseModel):
    conversation_id: Optional[str] = None
    user_response: Optional[str] = None
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    node_history: List[Dict[str, Optional[str]]] = Field(default_factory=list)

class QuizGeneratorAgentOutput(BaseModel):
    message_to_user: Optional[str] = Field(None, description="An optional message to the user")
    quiz: List[str] = Field(description="A list of questions and answers",min_items=2,max_items=3)

class AnswerCheckerAgentInput(BaseModel):
    conversation_id: Optional[str] = None
    user_response: Optional[str] = None
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    node_history: List[Dict[str, Optional[str]]] = Field(default_factory=list)

class AnswerCheckerAgentOutput(BaseModel):
    message_to_user: Optional[str] = Field(None, description="An optional message to the user")
    correct_answer: str = Field(description="The correct answer to the question")
    user_answer: str = Field(description="The user's answer to the question")
    is_correct: bool = Field(description="Whether the user's answer is correct")


# --- PydanticAI Agent Definitions ---

# Use a compatible LLM wrapper (e.g., from langchain)
llm_engine = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

welcome_agent = Agent(
    model="gpt-4o-mini",
    deps_type=WelcomeAgentInput,
    result_type=WelcomeAgentOutput,
    system_prompt="""
        You are a helpful AI assistant. Your name is Welcome Agent. You provide a welcome message to the user.
        You are part of a multi-agent system that generates a quiz based on the user's response.
        You also respond to chitchat and optionally handoff to a specialized agent based on the user's response.
        The specialized agents are:
        1. Quiz Generator Agent (return the agent_to_use as quiz_generator_agent) - This agent generates a quiz based on the user's response
        2. Answer Checker Agent (return the agent_to_use as answer_checker_agent) - This agent checks the user's answer against the correct answer
    """
)

quiz_generator_agent = Agent(
    model="gpt-4o-mini",
    deps_type=QuizGeneratorAgentInput,
    result_type=QuizGeneratorAgentOutput,
    system_prompt="""
    You are a helpful AI assistant. Your name is Quiz Generator Agent. You generate a quiz based on the user's response.
    The quiz should be in the form of a list of questions and answers.
    The questions should be in the form of a question and the answer should be in the form of a list of options.
   
    """
)

answer_checker_agent = Agent(
    model="gpt-4o-mini",
    deps_type=AnswerCheckerAgentInput,
    result_type=AnswerCheckerAgentOutput,
    system_prompt="""
    You are a helpful AI assistant. Your name is Answer Checker Agent. You check the user's answer against the correct answer.
    if the user's answer is correct, return "correct".
    if the user's answer is incorrect, return "incorrect".

    """
)

# --- Async PydanticAI Agent Runner Functions ---
# These functions remain largely the same, just ensuring they use the initialized agents
async def run_welcome_agent(input_data: dict) -> WelcomeAgentOutput:
    """Run the Welcome agent with the given input"""
    try:
        deps = WelcomeAgentInput(**input_data)
        
        result = await welcome_agent.run( # Using await assumes PydanticAI's run or underlying call is awaitable
            deps.user_response,
            deps=deps
        )
        print(f"DEBUG: Welcome Agent Raw Result: {result}")
        # PydanticAI >= 1.0 returns a result object, access data via .result or .data depending on version
        # Adjust based on your PydanticAI version. Assuming older version returning the model directly.
        if isinstance(result, WelcomeAgentOutput):
             return result
        elif hasattr(result, 'data') and isinstance(result.data, WelcomeAgentOutput):
             return result.data
        elif hasattr(result, 'result') and isinstance(result.result, WelcomeAgentOutput):
             return result.result
        else:
            raise TypeError(f"Unexpected result type from Welcome agent: {type(result)}")

    except Exception as e:
        print(f"Error details in run_welcome_agent: {e}")
        raise RuntimeError(f"Error running Welcome agent: {e}") from e

async def run_quiz_generator_agent(input_data: dict) -> QuizGeneratorAgentOutput:
    """Run the Quiz Generator agent with the given input"""
    try:
        deps = QuizGeneratorAgentInput(**input_data)
        result = await quiz_generator_agent.run(
            deps.user_response,
            deps=deps
        )
        print(f"DEBUG: Quiz Generator Agent Raw Result: {result}")
        return result
    except Exception as e:
        print(f"Error details in run_quiz_generator_agent: {e}")
        raise RuntimeError(f"Error running Quiz Generator agent: {e}") from e

async def run_answer_checker_agent(input_data: dict) -> AnswerCheckerAgentOutput:
    """Run the Answer Checker agent with the given input"""
    try:
        deps = AnswerCheckerAgentInput(**input_data)
        result = await answer_checker_agent.run( # Using await assumes PydanticAI's run or underlying call is awaitable
            deps.user_response,
            deps=deps
        )
        print(f"DEBUG: Answer Checker Agent Raw Result: {result}")
        # Handle different PydanticAI result structures
        if isinstance(result, AnswerCheckerAgentOutput):
             return result
        elif hasattr(result, 'data') and isinstance(result.data, AnswerCheckerAgentOutput):
             return result.data
        elif hasattr(result, 'result') and isinstance(result.result, AnswerCheckerAgentOutput):
             return result.result
        else:
            raise TypeError(f"Unexpected result type from Answer Checker agent: {type(result)}")

    except Exception as e:
        print(f"Error details in run_answer_checker_agent: {e}")
        raise RuntimeError(f"Error running Answer Checker agent: {e}") from e


    
# --- LangGraph State Definition ---
class AppState(BaseModel):
    user_response: Optional[str] = None
    conversation_id: Optional[str] = None
    messages: List[Dict[str, str]] = Field(default_factory=list) # Use Dict for JSON compatibility
    node_history: List[Dict[str, Optional[str]]] = Field(default_factory=list) # Track node outputs/errors
    agent_trace: List[str] = Field(default_factory=list)
    current_status: str = "started"
    welcome_agent_output: Optional[WelcomeAgentOutput] = None # Store the structured output
    quiz_generator_agent_output: Optional[QuizGeneratorAgentOutput] = None # Store the structured output
    answer_checker_agent_output: Optional[AnswerCheckerAgentOutput] = None # Store the structured output
    error_message: Optional[str] = None

    class Config:
        # Allow extra fields if necessary during runtime, though explicit updates are better
        extra = 'ignore'


# --- Refactored Async LangGraph Node Functions ---
# Nodes now MUST be async def and return a dictionary of state updates
async def welcome_node(state: AppState) -> Dict:
    """Async LangGraph node for the Welcome agent."""
    print("--- Entering Welcome Node ---")
    node_name = "welcome_agent"
    updated_trace = state.agent_trace + [node_name]
    updates: Dict[str, Any] = {"agent_trace": updated_trace}

    try:
        # Run the Welcome agent
        welcome_result = await run_welcome_agent({
            "user_response": state.user_response,
            "conversation_id": state.conversation_id,
            "conversation_history": state.messages,
            "node_history": state.node_history
        })
        
        # Update messages with agent's response
        updated_messages = state.messages.copy()
        if welcome_result.message_to_user:
            updated_messages.append({
                "role": "assistant",
                "content": welcome_result.message_to_user
            })
        
        updates.update({
            "welcome_agent_output": welcome_result,
            "messages": updated_messages,
            "current_status": "welcome_complete"
        })
        
        return updates
        
    except Exception as e:
        print(f"Error details in welcome_node: {e}")
        updates["error_message"] = str(e)
        return updates

async def quiz_generator_node(state: AppState) -> Dict:
    """Async LangGraph node for the Quiz Generator agent."""
    print("--- Entering Quiz Generator Node ---")
    node_name = "quiz_generator_agent"
    updated_trace = state.agent_trace + [node_name]
    updates: Dict[str, Any] = {"agent_trace": updated_trace}

    try:
        # Run the Quiz Generator agent
        quiz_generator_result = await run_quiz_generator_agent(
            {
                "user_response": state.user_response,
                "conversation_id": state.conversation_id,
                "conversation_history": state.messages,
                "node_history": state.node_history
            }
        )
        updates["quiz_generator_agent_output"] = quiz_generator_result
        #update conversation history if message_to_user is not None
        if quiz_generator_result.message_to_user:
            state.messages.append({
                "role": "assistant",
                "content": quiz_generator_result.message_to_user
            })
        return updates
    except Exception as e:
        print(f"Error details in quiz_generator_node: {e}")
        updates["error_message"] = str(e)
        updates["quiz_generator_agent_output"] = QuizGeneratorAgentOutput(message_to_user="An error occurred while processing your request. Please try again later.")
        #send error message to user
        state.messages.append({
            "role": "assistant",
            "content": "An error occurred while processing your request. Please try again later."
        })
        return updates

async def answer_checker_node(state: AppState) -> Dict:
    """Async LangGraph node for the Answer Checker agent."""    
    print("--- Entering Answer Checker Node ---")
    node_name = "answer_checker_agent"
    updated_trace = state.agent_trace + [node_name]
    updates: Dict[str, Any] = {"agent_trace": updated_trace}
    
    try:
        # Run the Answer Checker agent
        answer_checker_result = await run_answer_checker_agent(
            {
                "user_response": state.user_response,
                "conversation_id": state.conversation_id,
                "conversation_history": state.messages,
                "node_history": state.node_history
            }
        )
        updates["answer_checker_agent_output"] = answer_checker_result
        #update conversation history if message_to_user is not None
        if answer_checker_result.message_to_user:
            state.messages.append({
                "role": "assistant",
                "content": answer_checker_result.message_to_user
            })
        return updates
    except Exception as e:
        print(f"Error details in answer_checker_node: {e}")
        updates["error_message"] = str(e)
        updates["answer_checker_agent_output"] = AnswerCheckerAgentOutput(message_to_user="An error occurred while processing your request. Please try again later.")
        #send error message to user
        state.messages.append({
            "role": "assistant",
            "content": "An error occurred while processing your request. Please try again later."
        })
        return updates

async def welcome_node_decision(state: AppState) -> str:
    """Async LangGraph node for the Welcome agent decision."""
    print("--- Entering Welcome Node Decision ---")
    
    try:
        if state.welcome_agent_output:
            agent_to_use = state.welcome_agent_output.agent_to_use
            print(f"Routing to: {agent_to_use}")
            
            # Return string directly for routing
            if agent_to_use == "quiz_generator":
                return "quiz_generator_agent"
            elif agent_to_use == "answer_checker":
                return "answer_checker_agent"
            else:
                return "end"  # Return END directly if no specific agent
        else:
            print("No welcome agent output found")
            return "end"
            
    except Exception as e:
        print(f"Error in welcome node decision: {e}")
        return "end"
    
# --- LangGraph Graph Definition ---
# Use the async node functions
workflow = StateGraph(AppState)

workflow.add_node("welcome_agent", welcome_node) # Use the async node function
workflow.add_node("quiz_generator_agent", quiz_generator_node) # Use the async node function
workflow.add_node("answer_checker_agent", answer_checker_node) # Use the async node function

# Define the graph flow (same linear flow)
workflow.add_edge(START, "welcome_agent")
#add conditional edge   
workflow.add_conditional_edges(
    "welcome_agent",
    welcome_node_decision,
    {
        "quiz_generator_agent": "quiz_generator_agent",
        "answer_checker_agent": "answer_checker_agent",
        "end": END
    }
)
workflow.add_edge("quiz_generator_agent", END)
workflow.add_edge("answer_checker_agent", END)
# Compile the graph
# No checkpointer added in this refactor, as per instructions
compiled_graph = workflow.compile()

# --- Main Execution Block ---
async def main():
    """Asynchronous main function to run the graph."""
    messages = []  # Initialize conversation history
    
    while True:
        user_response = input("Enter your response: ")
        if user_response.lower() in ["exit", "quit"]:
            break
        
        # Add user message to history
        messages.append({
            "role": "user",
            "content": user_response
        })
        
        # Create initial state with existing messages
        initial_state = AppState(
            user_response=user_response,
            conversation_id=str(uuid.uuid4()),
            current_status="started",
            messages=messages,  # Pass the ongoing conversation
            node_history=[],
            agent_trace=[]
        )

        print("\n--- Invoking Graph ---")
        final_state = await compiled_graph.ainvoke(initial_state)
        print("\n--- Graph Execution Complete ---")

        #print the final state
        print(f"Final State: {final_state}")

   


if __name__ == "__main__":
    asyncio.run(main())