# main.py (Place this in the quiz-generator directory)

import logging
import os
import uuid
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from langgraph.graph.state import StateGraph
from langgraph.checkpoint.memory import MemorySaver

# Project specific imports - adjust paths to import from app directory
from app.graph.quiz_gen_graph_nf import build_graph
from app.models.pydantic_models import (
    QuizState,
    QuizGenerationRequest,
    QuizGenerationResponse
)

# --- Configuration Loading ---
load_dotenv()  # Load environment variables from .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Needed by agents potentially
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Warning if OpenAI key is missing, as agents might fail later
if not OPENAI_API_KEY or not GEMINI_API_KEY:
    print("Warning: OPENAI_API_KEY or GEMINI_API_KEY environment variable not set. LLM calls may fail.")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables ---
compiled_graph: Optional[StateGraph] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    Initializes and compiles the graph.
    """
    global compiled_graph
    compiled_graph = None
    logger.info("Application startup: Initializing resources...")

    # Use in-memory checkpointing for this simple script
    memory_saver = MemorySaver()

    # Build and compile the graph
    try:
        logger.info("Building LangGraph definition...")
        workflow = build_graph()
        logger.info("Compiling LangGraph...")
        compiled_graph = workflow.compile(checkpointer=memory_saver)
        logger.info("LangGraph compiled successfully with MemorySaver.")
    except Exception as e:
        logger.exception("FATAL: Failed to build or compile LangGraph.")
        raise RuntimeError("Failed to build or compile graph") from e

    yield

    logger.info("Application shutdown: Cleaning up resources...")
    logger.info("Cleanup finished.")

# --- FastAPI Application Instance ---
app = FastAPI(
    title="AI Quiz Generator API",
    description="API endpoint to interact with the LangGraph-based Quiz Generator",
    version="0.1.0",
    lifespan=lifespan
)

@app.post(
    "/invoke",
    summary="Invoke the Quiz Generation Agent Flow",
    tags=["Quiz Generation"]
)
async def invoke_agent_flow(request: QuizGenerationRequest = Body(...)):
    """
    Receives user input (text explanation) and an optional conversation ID.
    Invokes the LangGraph flow without persistent memory.
    Returns the latest status, message for the user, and conversation ID.
    """
    global compiled_graph

    if compiled_graph is None:
        logger.error("Graph not compiled during startup. Cannot process request.")
        raise HTTPException(status_code=500, detail="Internal Server Error: Graph not ready.")

    conversation_id = request.conversation_id or f"session_{uuid.uuid4()}"
    logger.info(f"Processing request for conversation_id: {conversation_id}")

    input_state: QuizState = {
        "user_input": request.explanation,
        "conversation_id": conversation_id,
    }

    try:
        logger.info(f"Invoking graph for conversation_id: {conversation_id}...")
        final_state = await compiled_graph.ainvoke(input=input_state)
        logger.info(f"Graph invocation complete for conversation_id: {conversation_id}")

    except Exception as e:
        logger.exception(f"Error during graph invocation for conversation_id: {conversation_id}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error during graph execution: {str(e)}")

    if not final_state or not isinstance(final_state, dict):
        logger.error(f"Graph invocation for {conversation_id} did not return a valid final state dictionary.")
        raise HTTPException(status_code=500, detail="Internal Server Error: Invalid final state.")

    last_agent_output = final_state.get('last_agent_output', {})
    message_to_user = last_agent_output.get('message_to_user', "Processing complete.")
    error_message = final_state.get('error_message')

    response_status = "error" if error_message or last_agent_output.get('status') == 'error' else "completed"

    response_data = QuizGenerationResponse(
        conversation_id=conversation_id,
        status=response_status,
        message=error_message or message_to_user,
        questions=[]
    )

    logger.info(f"Sending response for conversation_id: {conversation_id}, Status: {response_status}")
    return response_data
   

@app.get("/", tags=["Health Check"])
async def read_root():
    return {"message": "AI Quiz Generator API is running."}

async def run_quiz_generator():
    """Run the quiz generator application."""
    # Initialize with welcome message
    initial_state = QuizState(
        user_input=None,
        topic=None,
        questions=[],
        error_message=None,
        current_step="welcome",
        response_to_user=None,
        welcome_attempts=0,
        message_to_user="Welcome to the Quiz Generator! What topic would you like a quiz on? (Type 'exit' to quit)",
        conversation_history=[],
        node_history=[]
    )
    
    # Build and compile the graph
    workflow = build_graph()
    app = workflow.compile()
    
    # Run the graph
    print("\n--- Invoking Graph ---\n")
    result = await app.ainvoke(initial_state.model_dump())
    print("\n--- Graph Execution Finished ---\n")

def main():
    """Main entry point for the application."""
    try:
        # Initialize the graph using the existing FastAPI app
        async def init_graph():
            async with lifespan(app):
                await run_quiz_generator()
        
        # Run the initialization and quiz generator
        asyncio.run(init_graph())
    except KeyboardInterrupt:
        print("\nAssistant: Exiting Quiz Generator.")
    except Exception as e:
        print(f"Fatal error: {e}")

if __name__ == "__main__":
    main()