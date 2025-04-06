from typing import List, Optional, Dict, Any, TypedDict

# Import the base output model and potentially specific ones if needed for type hints
from app.models.pydantic_models import BaseAgentOutput, WelcomeAgentOutput, QuizGenerationAgentOutput, NodeOutput

class AppState(TypedDict, total=False):
    """
    Represents the state of the quiz generation process in the LangGraph,
    using explicit handoffs and node history.
    """

    # --- Input & Context ---
    input_explanation: str              # User's current input message content
    conversation_id: str              # Unique ID for checkpointer/session tracking

    # --- Episodic Memory (Managed by Checkpointer) ---
    messages: List[Dict[str, str]]      # Append-only list {"role": "user/assistant", "content": ...}

    # --- Working Memory / Process Flow ---
    agent_trace: List[str]              # Simple list of agent node names executed
    current_status: str               # Overall status (e.g., "processing", "awaiting_handoff", "completed", "error")
    error_message: Optional[str]      # Stores critical error messages if they occur

    # --- Agent Decision History & Handoff ---
    # Stores the structured output of each agent node after execution
    node_history: List[NodeOutput]      # List of BaseAgentOutput compatible dicts
    # Store the *last* agent's output object for easy access by routing logic
    # Note: Using Dict here, but it represents a BaseAgentOutput structure
    last_agent_output: Optional[Dict[str, Any]]

    # --- LLM / Tool Call Related ---
    llm_calls: int                      # Counter for LLM interactions

    # --- Potential Future / Intermediate ---
    # generated_quiz_object: Optional[Quiz] # Maybe store the final quiz here too