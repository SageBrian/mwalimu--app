from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union, Literal, Dict, Any, Annotated
from datetime import datetime

# Data Structures for MwalimuBot

# Langgraph main state
class MwalimuBotState(BaseModel):
    user_input: str
    user_id:str
    phone_number:str
    first_node:str
    conversation_history: List[Dict[str, Any]] 
    node_history: List[Dict[str, Any]]
    message_to_student: Optional[str] = None
    current_subject: Optional[str] = None
    current_grade: int = Field(default=0, ge=0, le=12)
    rag_context: Optional[str] = None
    ready_for_tutoring: bool = False
    ready_for_quiz: bool = False
    handoff_agents: List[str] = Field(default=[])
    handoff_agents_params: List[Dict[str, Any]] = Field(default=[])
    router_attempts: int = 0
    tutor_attempts: int = 0
    error_message: Optional[str] = None
    current_step: Optional[str] = None
    response_to_user_attempts: int = 0

class Student(BaseModel):
    id: str
    name: Optional[str] = None
    subjects: Optional[List[str]] = None
    current_subject: Optional[str] = None
    current_grade: int = Field(default=0, ge=0, le=12)
    router_attempts:int = 0

class Question(BaseModel):
    id: str
    question: str
    options: List[str]
    answer: str
    
class Quiz(BaseModel):
    id: str
    subject: str
    grade: int
    questions: List[Question]


class TutorParameters(BaseModel):
    subject: str
    grade: int

    

class RespondToUserParameters(BaseModel):
    message_to_student: str
    agent_after_response: str = "routing_agent"

class HandoffParameters(BaseModel):
    """Parameters for handoff to agents."""
    agent_name: Literal["respond_to_user", "tutor_agent"] = Field(description="Name of the agent to handoff to, must be another agent")
    message_to_agent: str = Field(description="Message  to the agent to help it understand the request")
    agent_specific_parameters: Union[TutorParameters, RespondToUserParameters]= Field("agent specific parameters i.e TutorParameters for tutor_agent or RespondToUserParameters for respond_to_user")
    


class Handoff(BaseModel):
    """Main handoff model with a list of agents"""
    """
    This model is used to handoff to agents.
    It contains a list of agents to handoff to.
    Each agent has a name, a message to the agent, and agent specific parameters.
    """
    handoff_agents: List[HandoffParameters] = Field(description="List of agents to handoff to")
    
class ChatRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="User ID")
    phone_number: str = Field(description="Phone Number")
    message: str = Field(description="Message")

class ChatResponse(BaseModel):
    phone_number: str = Field(description="Phone Number")
    message: str = Field(description="Message")
    error: Optional[str] = Field(None, description="Error")

