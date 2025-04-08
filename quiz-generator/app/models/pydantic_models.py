from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union, Literal, Dict, Any, Annotated
from datetime import datetime

# --- Data Structures for Quiz ---
class QuizQuestion(BaseModel):
    question: str = Field(description="The quiz question")
    # Ensuring options have clear labels for display/answering
    options: List[str] = Field(description="List of possible answers, prefixed like 'A. Answer Text'")
    correct_answer: str = Field(description="The correct answer (matching one of the options exactly)")
    explanation: str = Field(description="Explanation for the answer")

# --- API Request/Response Models ---
class QuizGenerationRequest(BaseModel):
    """Request model for quiz generation endpoint."""
    explanation: str = Field(description="User's explanation or topic for quiz generation")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for tracking")

class QuizGenerationResponse(BaseModel):
    """Response model for quiz generation endpoint."""
    conversation_id: str = Field(description="Conversation ID for tracking")
    status: str = Field(description="Status of the quiz generation (completed/error)")
    message: str = Field(description="Message to display to the user")
    questions: List[QuizQuestion] = Field(default_factory=list, description="Generated quiz questions")

# --- Simplified LLM Response Schemas (Removed redundant current_step) ---
class WelcomeResponse(BaseModel):
    """Response schema for welcome logic."""
    response_to_user: str = Field(description="Response to show to the user")
    topic: Optional[str] = Field(None, description="Extracted topic if found")

class QuizResponse(BaseModel):
    """Response schema for quiz generator agent."""
    questions: List[QuizQuestion] = Field(description="List of generated quiz questions")

#Handoff to agents
class RespondToUserParameters(BaseModel):
    message_to_user: str = Field(..., description="Message to display to the user")
    agent_after_response: Optional[str] = Field(None, description="Agent to handoff to after response from user, default is the agent handing off")

class QuizGenParameters(BaseModel):
    topic: str = Field(..., description="User's explanation or topic for quiz generation")
    difficulty: Optional[str] = Field("medium", description="The difficulty level of the quiz, default is medium")
    num_questions: Optional[int] = Field(5, description="The number of questions in the quiz, default is 5")
    tone: Optional[str] = Field("neutral", description="The tone of the quiz, default is neutral")

class HandoffParameters(BaseModel):
    """Parameters for handoff to agents."""
    agent_name: Literal["question_generator", "respond_to_user"] = Field(description="Name of the agent to handoff to, must be either 'question_generator' or 'respond_to_user'")
    message_to_agent: str = Field(description="Message  to the agent to help it understand the request")
    agent_specific_parameters: Union[RespondToUserParameters, QuizGenParameters] = Field(description="Agent specific parameters")


# --- Refined LangGraph State ---
class QuizState(BaseModel):
    """Represents the state of the quiz generation process."""
    user_input: Optional[str] = None
    quiz_parameters: Optional[Dict[str, Any]] = None
    questions: List[QuizQuestion] = []
    error_message: Annotated[Optional[str], Field(default=None)] = None
    current_step: Annotated[str, Field(default="welcome")]
    response_to_user: Optional[str] = None
    welcome_attempts: int = 0
    message_to_user: Optional[str] = None
    conversation_history: List[Dict[str, str]] = []
    topic: Optional[str] = None
