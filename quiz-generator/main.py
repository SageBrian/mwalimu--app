# main.py - Simplified for human-in-the-loop conversation handling

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uuid
import logging
from dotenv import load_dotenv

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import Pydantic Models
from app.models.pydantic_models import (
    MwalimuBotState,
    ChatRequest
)

# Import DB Functions
from app.shared_services.save_load_conversation import save_conversation, load_conversation

# Import Graph Builder
from app.graph.graph import build_graph

# Build and compile the graph once when the application starts
workflow = build_graph()
graph = workflow.compile()
logger.info("LangGraph built and compiled successfully.")

# FastAPI App Instance
app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "MwalimuBot is running!"}

@app.post("/chat/")
async def chat_endpoint(request: ChatRequest):
    """
    Handles chat interactions with human-in-the-loop state management.
    """
    user_id = request.user_id
    user_message = request.message
    loaded_state = None

    print(f"User ID: {user_id}")
    print(f"User Message: {user_message}")

    try:
        # 1. Handle existing conversation or create new one
        phone_number = request.phone_number 
        
        # Try to load existing conversation if ID was provided
        if phone_number:
            loaded_state_data = load_conversation(phone_number)
            if loaded_state_data:
                # Convert loaded data to QuizState and update with new message
                loaded_state = MwalimuBotState.model_validate(loaded_state_data)
                # Reset temporary state fields
                loaded_state.message_to_student = None
                loaded_state.error_message = None
                loaded_state.handoff_agents_params = []
                loaded_state.handoff_agents = []
                # Add new message
                loaded_state.user_input = user_message
                loaded_state.conversation_history.append({
                    "role": "human", 
                    "content": user_message
                })
    
                logger.info(f"Loaded and updated existing state for conversation {phone_number}")

        # Create new state if none exists or wasn't found
        if not loaded_state:
            # Initialize new state with all required fields
            student_id = str(uuid.uuid4())
            loaded_state = MwalimuBotState(
                user_id=user_id,
                phone_number=phone_number,
                user_input=user_message,
                conversation_history=[{"role": "human", "content": user_message}],
                current_subject=None,
                current_grade = 2,
                rag_context = None,
                node_history=[],
                ready_for_tutoring= False,
                ready_for_quiz = False,
                first_node="routing_agent",
                current_step=None,
                response_to_user_attempts=0
            )
            logger.info(f"Created new state for conversation {phone_number}")

        # 3. Run the graph with the state
        final_state = await graph.ainvoke(loaded_state.model_dump())
        final_state = MwalimuBotState.model_validate(final_state)
        logger.info(f"Graph execution completed for {phone_number}")

        # 4. Save the final state
        state_to_save = final_state.model_dump()
        state_to_save["phone_number"] = phone_number  # Ensure phone_number is in the state
        save_conversation(state_to_save)
        logger.info(f"Saved state for conversation {phone_number}")

        # 5. Prepare response
        response_message = None
        if final_state.message_to_student:
            response_message = final_state.message_to_student
        elif final_state.response_to_user:
            response_message = final_state.response_to_user
        else:
            response_message = {
                "message_to_student": "Processing complete.",
               
            }

        # 6. Return response
        return {
            "response": response_message,
            "phone_number": phone_number
        }

    except Exception as e:
        logger.error(f"Error processing conversation {phone_number}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing conversation: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)