from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union, Literal, Dict, Any
from datetime import datetime

# --- Core Domain Models (Example - refine as needed) ---
class Question(BaseModel):
    text: str
    options: List[str]
    correct_option_index: int
    question_type: str  # Changed from having a default value

class Quiz(BaseModel):
    title: str
    description: Optional[str] = None
    questions: List[Question]  # Removed default_factory

# --- Input/Output for Specific Agents (Used within Handoffs) ---

# Welcome Agent doesn't really need a dedicated Input model for handoff
# as it's the entry point, getting data from AppState directly.

class QuizGenerationRequest(BaseModel):
    """Request model for quiz generation"""
    explanation: str
    conversation_id: Optional[str] = None

class QuizQuestion(BaseModel):
    """Model for a single quiz question"""
    question: str
    options: List[str]
    correct_answer: str
    explanation: Optional[str] = None

class QuizGenerationResponse(BaseModel):
    """Response model for quiz generation"""
    conversation_id: str
    status: str  # "completed" or "error"
    message: str
    questions: List[QuizQuestion] = []

class QuizGenerationAgentInput(BaseModel):
    """Input model for quiz generation agent"""
    input_explanation: str
    conversation_id: str

class QuizAgentInput(BaseModel):
    """Input parameters for an agent interacting with an existing quiz (e.g., taking the quiz)."""
    quiz: Quiz # The specific quiz being interacted with
    user_response: Optional[str] = Field(None, description="The user's answer or interaction related to the quiz.")
    # conversation_history: List[Dict[str,str]]

# --- Handoff Parameter Structures ---

class WelcomeHandoffParameters(BaseModel):
    # This seems less useful as Welcome is usually the start. Included for completeness if needed.
    agent_name: Literal["welcome_agent"] = "welcome_agent"
    # No specific agent_parameters needed as it reads from AppState

class QuizGenerationHandoffParameters(BaseModel):
    """Specifies handoff to the agent that *creates* quizzes."""
    agent_name: Literal["quiz_generation_agent"] = "quiz_generation_agent"
    agent_parameters: QuizGenerationAgentInput = Field(..., description="Parameters needed to generate the quiz.")

class QuizInteractionHandoffParameters(BaseModel):
    """Specifies handoff to the agent that manages *interaction* with an existing quiz."""
    agent_name: Literal["quiz_interaction_agent"] = "quiz_interaction_agent"
    agent_parameters: QuizAgentInput = Field(..., description="Parameters for interacting with the quiz.")

# Union type for agent-specific handoff parameters
AgentSpecificHandoffParams = Union[
    WelcomeHandoffParameters,
    QuizGenerationHandoffParameters,
    QuizInteractionHandoffParameters
    # Add other agent handoff parameter types here
]

class SingleHandoff(BaseModel):
    """Defines the parameters and context for handing off to a single agent."""
    parameters: AgentSpecificHandoffParams = Field(..., description="The specific agent and its input parameters.")
    reason: str = Field(..., description="Concise reason for this handoff.")
    message_to_agent: str = Field(..., description="Instruction or context for the next agent.")

    # Add a validator to ensure parameters match the name if needed, though Literal helps
    @field_validator('parameters')
    def check_parameters_match_name(cls, v):
        # Example validation (optional)
        if isinstance(v, QuizGenerationHandoffParameters) and v.agent_name != "quiz_generation_agent":
             raise ValueError("Mismatch between parameters type and agent_name")
        # Add checks for other types...
        return v

# MultiHandoff might be complex for now, let's focus on SingleHandoff
# class MultiHandoff(BaseModel):
#     handoffs: List[SingleHandoff]
#     reason: str = Field(..., description="Reason for multiple handoffs")

# --- Agent Output Models (Defining what each agent returns) ---

class BaseAgentOutput(BaseModel):
    """Common structure for agent outputs."""
    message_to_user: str
    handoff: Optional[SingleHandoff]
    agent_name: str
    status: str
    error_details: Optional[str]

class WelcomeAgentOutput(BaseModel):
    """Specific output structure for the Welcome Agent."""
    message_to_user: str
    detected_intent: str
    handoff: Optional[SingleHandoff]
    status: str
    error_details: Optional[str]
    agent_name: str

class QuizGenerationAgentOutput(BaseModel):
    """Output from the agent that generates quizzes."""
    message_to_user: str
    generated_quiz: Optional[Quiz]
    rationale: Optional[str]
    handoff: Optional[SingleHandoff]
    status: str
    error_details: Optional[str]
    agent_name: str

# Add QuizInteractionAgentOutput etc. as needed

# Generic type for node history entries (can be any agent's output)
NodeOutput = Dict[str, Any] # Store as dict for JSON compatibility with checkpointer

class KaribuAgentInput(BaseModel):
    """Input model for Karibu Agent"""
    conversation_id: Optional[str] = None
    user_response: Optional[str] = None

class KaribuAgentOutput(BaseModel):
    """Output model for Karibu Agent"""
    message_to_user: str

