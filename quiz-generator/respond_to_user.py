# quiz-generator/app/agents/respond_to_user.py

from typing import Dict, Any
from langchain_core.messages import HumanMessage
from app.models.pydantic_models import QuizState

async def respond_to_user_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node for handling user communication and updating conversation history."""
    # Convert state dict to QuizState if it's not already
    print("===== Entering Respond to User Node ======")
  
    current_state = state if isinstance(state, QuizState) else QuizState(**state)
    
    # Display message to user if there is one
    if current_state.message_to_user:
        print(f"\nAssistant: {current_state.message_to_user}")
        # Add assistant's message to conversation history
        current_state.conversation_history.append({
            "role": "assistant",
            "content": current_state.message_to_user
        })
        # Clear the message after displaying
        current_state.message_to_user = None
    
    # Get user input
    user_input = input("\nYou: ").strip()
    
    # Update state with user input
    current_state.user_input = user_input
    
    # Update conversation history with user input
    if user_input:
        current_state.conversation_history.append({
            "role": "user",
            "content": user_input
        })
    current_state.current_step = "welcome"
    print("===== Exiting Respond to User Node ======")
    
    return current_state.dict()