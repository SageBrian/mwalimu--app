"""
Graph Builder Module for No Frameworks
"""
from typing import Dict, Any, Literal


from app.models.pydantic_models import QuizState

#Nodes
from app.agents.welcome_agent_nf import welcome_node
from app.agents.respond_to_user import respond_to_user_node
from app.agents.question_generator import question_generator_node
from langgraph.graph import StateGraph, START, END

""" Graph starts with the welcome node ---
 After that, it checks handoff parameters to determine next node(s)
 Incase of multiple nodes, it will call the appropriate node(s) in the order of the handoff parameters.
 Ends with closure parameters"""


#--- Build the graph - with no frameworks ---
def decision_to_end(state: Dict[str, Any]) -> Dict[str, Any]:
    """Decision to end the graph."""
    current_state = state if isinstance(state, QuizState) else QuizState(**state)
    
    if current_state.current_step == "welcome":
        return "welcome"
    elif current_state.current_step == "question_generator":
        return "question_generator"
    else:
        return "end"
def decision_after_welcome(state: Dict[str, Any]) -> Dict[str, Any]:
    print("===== Entering Decision After Welcome Node ======")
    """Decision after welcome.- check handoff agents parameters"""
    current_state = state if isinstance(state, QuizState) else QuizState(**state)
    # Get handoff agents parameters
    handoff_agents = current_state.handoff_agents
    # Check if handoff agents parameters is not empty
    print(f"===== Handoff Agents: {handoff_agents} =====")
    if handoff_agents:
        # Check if the first agent is respond_to_user
        if handoff_agents[0] == "respond_to_user":
            return "respond_to_user"
        elif handoff_agents[0] == "question_generator":
            return "question_generator"
        else:
            return "end"
    else:
        return "end"
    print("===== Exiting Decision After Welcome Node ======")
   
    
    
   

def build_graph(state: Dict[str, Any] = None) -> Dict[str, Any]:
    """Build the quiz generation workflow graph."""
    #--- Start with the welcome node ---
    workflow = StateGraph(QuizState)

    workflow.add_node("welcome", welcome_node)
    workflow.add_node("respond_to_user", respond_to_user_node)
    workflow.add_node("question_generator", question_generator_node)

    #--- Add edges ---
    workflow.add_edge(START, "respond_to_user")
    workflow.add_edge("respond_to_user", "welcome")
    workflow.add_conditional_edges(
        "welcome",
        decision_after_welcome,
        {
            "respond_to_user": "respond_to_user",
            "question_generator": "question_generator",
            "end": END}
    )

    workflow.add_conditional_edges(
        "respond_to_user",
        decision_to_end,
        {
            "welcome": "welcome",
            "question_generator": "question_generator",
            "end": END
        }
    )

    return workflow


    
