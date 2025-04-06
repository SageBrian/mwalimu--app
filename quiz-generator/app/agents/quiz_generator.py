# --- Import Libraries ---
import asyncio
import uuid
import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import json

# --- Load Environment Variables ---
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY not found in environment variables.")
    # Consider exiting or raising an error if the key is essential
    # exit()

# --- PydanticAI and Langchain Imports ---
# Assuming PydanticAI v1.x - Adjust if using older versions

try:
    from pydantic_ai import Agent as PydanticAIAgent # Renamed for clarity
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.models.gemini import GeminiModel
   
except ImportError:
    print("Please install pydantic-ai: pip install pydantic-ai")
    exit()
# Using OpenAI directly from pydantic-ai if available, or fallback
#llm_engine = OpenAIModel( "gpt-4o-mini", provider=OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY")))
llm_engine = GeminiModel( "gemini-2.0-flash", provider='google-gla')
#from pydantic_ai.models import OpenAI
from pydantic_ai import Agent as PydanticAIAgent
# LangGraph imports
try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.errors import GraphRecursionError # For potential loop handling later
except ImportError:
    print("Please install langgraph: pip install langgraph")
    exit()

# --- Pydantic Models (Agent I/O, State, Quiz Structure) ---

# --- Data Structures for Quiz ---
class QuizQuestion(BaseModel):
    question: str = Field(description="The quiz question")
    options: List[str] = Field(description="List of possible answers")
    correct_answer: str = Field(description="The correct answer")
    hint_correct_answer: Optional[str] = Field(None, description="Optional hint for the correct answer")
    explanation_correct_answer: Optional[str] = Field(None, description="Optional explanation for why the answer is correct")
    explanation_incorrect_answers: Optional[List[str]] = Field(None, description="Optional explanations for incorrect answers")

# --- Agent Input/Output Schemas ---
class WelcomeAgentInput(BaseModel):
    user_response: Optional[str] = None
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)

class QuizGeneratorParameters(BaseModel):
    topic: str = Field(description="The topic of the quiz to generate.")
    num_questions: int = Field(description="The number of questions to generate.", default=3)
    difficulty: str = Field(description="The difficulty level of the quiz.", default="medium")
    message_to_agent: Optional[str] = Field(None, description="A message to the agent to help it understand the user's request better.")
    
class WelcomeAgentOutput(BaseModel):
    message_to_user: Optional[str] = Field(None, description="A greeting, response to chitchat, or prompt for the next step.")
    agent_to_use: Optional[str] = Field(None, description="The agent to route to next ('quiz_generator_agent', 'answer_checker_agent') or None if just chatting/ending.", choices=["quiz_generator_agent", "answer_checker_agent"])
    quiz_parameters: Optional[QuizGeneratorParameters] = Field(None, description="Parameters for the quiz generation., To be used if next agent is 'quiz_generator_agent'")
   


class QuizGeneratorAgentInput(BaseModel):
    user_response: Optional[str] = None
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    quiz_parameters: Optional[Dict[str, Any]] = None  # Changed to Dict type

class QuizGeneratorAgentOutput(BaseModel):
    message_to_user: Optional[str] = Field(None, description="A message to the user, seeking clarification if needed or anything else to help the user understand the quiz better.")
    quiz: Optional[List[QuizQuestion]] = Field(description="A list of quiz questions, each with text, options, and the correct answer.", min_items=1, max_items=5) # Adjusted min/max items
 
class AnswerCheckerAgentInput(BaseModel):
    user_response: str # The user's answer attempt
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    # Context needed for checking:
    question_text: str = Field(description="The specific question the user was answering.")
    correct_answer: str = Field(description="The correct answer to the question.")

class AnswerCheckerAgentOutput(BaseModel):
    message_to_user: str = Field(description="Feedback to the user (e.g., 'Correct!', 'Incorrect, the answer was X').")
    is_correct: bool = Field(description="Whether the user's answer matches the correct answer.")
    user_answer: str # Store what the agent interpreted as the user's answer
    correct_answer: str # Reiterate the correct answer

# --- PydanticAI Agent Definitions ---

welcome_agent = PydanticAIAgent(
    model=llm_engine,
    deps_type=WelcomeAgentInput,
    result_type=WelcomeAgentOutput,
    system_prompt="""

        You are the first agent in maswali, a multi-agent system that generates quizzes based on user requests.
        You are responsible for collecting the user's request and routing it to the appropriate agent based on the user's request.

        Current-context:
        user_response {user_response}: The user's response to the welcome agent.
        conversation_history {conversation_history}: The conversation history between the user and the assistant.

        NOTE: It is very important to scan and fully understand the user's message and conversation history before responding.
        This is crucial for a natural interaction, and to avoid asking the same questions multiple times.



        Output Format:
        - You MUST Respond either with a message to the user or with the agent to route to next, or both.
        - Adhere strictly to the 'WelcomeAgentOutput' schema.
        - The 'message_to_user' field must be a string.
        - The 'agent_to_use' field must be a string, either 'quiz_generator_agent' or 'answer_checker_agent'.

        RULES:
        1. Understand the user's message & conversation history.
        2. If you can answer the user's question directly, do so. This is especially important for chitchat.
        3. Make the interaction as natural as possible. Be sure to introduce yourself and nudge the user towards generating a quiz.
        4. Try to collect as much information as possible about the quiz parameters from the user's message. These include:
            - Topic:(e.g., "Nairobi", "Kenya")
            - Number of questions: Any number (e.g., "5", "three")
            - Difficulty: Any difficulty level (e.g., "easy", "medium", "hard")
        5. Make the collection of quiz parameters as natural as possible. For example, you can ask the user to specify the number of questions, or to specify the difficulty level.
        6. Remember, the quiz parameters are optional. The user may not always specify them.
        7. If you have all the information you need, route to the quiz_generator_agent with the quiz parameters.

        DO NOT ROUTE TO THE QUIZ GENERATOR AGENT IF YOU DO NOT THE INFORMATION BELOW:
        - Topic (Mandatory)
        - Message to agent (Optional, to help the quiz_generator_agent understand the user's request better)
        - Number of questions (Optional, default is 3)
        - Difficulty (Optional, default is "medium")

    """
)

quiz_generator_agent = PydanticAIAgent(
    model=llm_engine,
    deps_type=QuizGeneratorAgentInput,
    result_type=QuizGeneratorAgentOutput,
    system_prompt="""
        You are the 'Quiz Generator Agent'. Your task is to create a short multiple-choice quiz based on below parameters
        user_response {user_response}: The user's response to the welcome agent.
        quiz_parameters {quiz_parameters}: The parameters for the quiz generation.
        conversation_history {conversation_history}: The conversation history between the user and the assistant.
        past_quiz_questions {past_quiz_questions}: The past quiz questions generated by the quiz_generator_agent.

        Output Format:
        - Adhere strictly to the 'QuizGeneratorAgentOutput' schema.
        - The 'quiz' field must be a JSON list, where each item is an object matching the 'QuizQuestion' schema: 
        {"question": "...", "options": ["A", "B", "C"], "correct_answer": "...", "hint_correct_answer": "...", "explanation_correct_answer": "...", "explanation_incorrect_answers": ["..."]}.
        - The 'correct_answer' MUST be one of the strings present in the 'options' list.
        - Generate between all questions requested, in batches of 5.
        - Optionally, provide a 'message_to_user' field, which could introduce the quiz or present the very first question from the generated 'quiz' list (e.g., "Here's your first question:\n{quiz[0].question}\nOptions: {', '.join(quiz[0].options)}"). If you include the first question in the message, make it clear.
    
        Example valid outputs:
        {
            "message_to_user": "I need more information. What topic would you like the quiz about?",
            "quiz": null
        }
        OR
        {
            "message_to_user": "Here's your quiz about Kenya!",
            "quiz": [{"question": "...", "options": [...], "correct_answer": "..."}]
        }
    """
    #description="Generates a multiple-choice quiz based on a user-provided topic."
)


answer_checker_agent = PydanticAIAgent(
    model=llm_engine,
    deps_type=AnswerCheckerAgentInput,
    result_type=AnswerCheckerAgentOutput,
    system_prompt="""
        You are the 'Answer Checker Agent'. You receive the user's response (their attempted answer), the original question text, and the correct answer text.
        Your tasks are:
        1. Compare the 'user_response' to the provided 'correct_answer'. Perform a reasonably flexible comparison (e.g., ignore case, minor typos if obvious).
        2. Determine if the user's answer is correct ('is_correct' = True) or incorrect ('is_correct' = False).
        3. Populate the 'AnswerCheckerAgentOutput' schema accurately:
            - 'is_correct': Boolean result of the comparison.
            - 'user_answer': The user's response received in the input.
            - 'correct_answer': The correct answer received in the input.
            - 'message_to_user': Provide feedback.
                - If correct: "Correct!", "Well done!", etc.
                - If incorrect: "Sorry, that's incorrect. The correct answer was: [correct_answer]."
    """,
    #description="Checks if the user's answer to a specific quiz question is correct."
)

# --- Async PydanticAI Agent Runner Functions ---
async def run_welcome_agent(input_data: dict) -> WelcomeAgentOutput:
    """Runs the Welcome agent asynchronously."""
    print(f"DEBUG: Running Welcome Agent with input: {input_data}")
    try:
        # Only pass the fields that the model expects
        input_model = WelcomeAgentInput(
            user_response=input_data.get('user_response'),
            conversation_history=input_data.get('conversation_history', [])
        )
        
        result = await welcome_agent.run(
            input_model.user_response or "",  # Primary input
            deps=input_model  # Dependencies
        )
        
        return result.data if hasattr(result, 'data') else result
        
    except Exception as e:
        print(f"ERROR in run_welcome_agent: {e}")
        return WelcomeAgentOutput(
            message_to_user="Sorry, I encountered an issue. Could you please repeat that?",
            agent_to_use=None
        )

async def run_quiz_generator_agent(input_data: dict) -> QuizGeneratorAgentOutput:
    """Runs the Quiz Generator agent asynchronously."""
    print(f"DEBUG: Running Quiz Generator Agent with input: {input_data}")
    try:
        # Convert quiz_parameters to dict if it's a Pydantic model
        quiz_params = input_data.get('quiz_parameters')
        if hasattr(quiz_params, 'model_dump'):
            quiz_params = quiz_params.model_dump()
        
        # Create input model with dict parameters
        input_model = QuizGeneratorAgentInput(
            user_response=input_data.get('user_response'),
            conversation_history=input_data.get('conversation_history', []),
            quiz_parameters=quiz_params,
            past_quiz_questions=input_data.get('past_quiz_questions', [])
        )
        
        result = await quiz_generator_agent.run(
            input_model.user_response or "",  # Primary input
            deps=input_model  # Dependencies
        )
        
        output = result.data if hasattr(result, 'data') else result
        print(f"DEBUG: Quiz Generator Agent Raw Result: {output}")
        
        return output
        
    except Exception as e:
        print(f"ERROR in run_quiz_generator_agent: {e}")
        return QuizGeneratorAgentOutput(
            message_to_user="Sorry, I encountered an issue generating the quiz. Here's a simple question instead.",
            quiz=[QuizQuestion(
                question="What is the capital city of Kenya?",
                options=["Nairobi", "Mombasa", "Kisumu", "Nakuru"],
                correct_answer="Nairobi"
            )]
        )

async def run_answer_checker_agent(input_data: dict) -> AnswerCheckerAgentOutput:
    """Runs the Answer Checker agent asynchronously."""
    print(f"DEBUG: Running Answer Checker Agent with input: {input_data}")
    try:
        # Ensure required context is present
        if 'question_text' not in input_data or 'correct_answer' not in input_data:
            raise ValueError("Answer Checker requires 'question_text' and 'correct_answer' in input_data.")

        input_model_instance = AnswerCheckerAgentInput(**input_data)
        result = await answer_checker_agent.run(input_model_instance)
        print(f"DEBUG: Answer Checker Agent Raw Result: {result}")

        if isinstance(result, AnswerCheckerAgentOutput):
             return result
        else:
             # Handle potential nested structure or parse
            if hasattr(result, 'data') and isinstance(result.data, AnswerCheckerAgentOutput):
                 return result.data
            elif hasattr(result, 'result') and isinstance(result.result, AnswerCheckerAgentOutput):
                 return result.result
            else:
                try:
                    if isinstance(result, str): result = json.loads(result)
                    if isinstance(result, dict): return AnswerCheckerAgentOutput(**result)
                except (json.JSONDecodeError, TypeError, ValueError):
                     raise TypeError(f"Unexpected/unparseable result type from Answer Check agent: {type(result)}, Content: {result}")
            raise TypeError(f"Unexpected result type from Answer Checker agent: {type(result)}")

    except Exception as e:
        print(f"ERROR in run_answer_checker_agent: {e}")
        # Fallback: Generate an error message adhering to schema
        return AnswerCheckerAgentOutput(
            message_to_user=f"Sorry, I couldn't check your answer due to an error: {e}",
            is_correct=False, # Default to false on error
            user_answer=input_data.get("user_response", "Unknown"),
            correct_answer=input_data.get("correct_answer", "Unknown")
        )


# --- LangGraph State Definition ---
class AppState(BaseModel):
    """Represents the overall state of the conversation and quiz."""
    user_response: Optional[str] = None
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Dict[str, str]] = Field(default_factory=list) # Stores user/assistant messages
    agent_trace: List[str] = Field(default_factory=list) # Tracks executed nodes
    current_status: str = "started"
    error_message: Optional[str] = None
    quiz_parameters: Optional[QuizGeneratorParameters] = None  # Add this field

    # Agent-specific outputs / state
    welcome_agent_output: Optional[WelcomeAgentOutput] = None
    quiz_generator_agent_output: Optional[QuizGeneratorAgentOutput] = None
    answer_checker_agent_output: Optional[AnswerCheckerAgentOutput] = None

    # Quiz-specific state
    quiz_questions: Optional[List[QuizQuestion]] = None # Stores the generated quiz
    current_question_index: int = Field(default=-1) # Index of the question being currently asked (-1 if no quiz active/finished)
    past_quiz_questions: Optional[List[QuizQuestion]] = None # Stores the past quiz questions

    class Config:
        extra = 'ignore' # Ignore extra fields during model initialization

# --- Async LangGraph Node Functions ---
# Nodes now MUST be async def and return a dictionary of state updates

async def welcome_node(state: AppState) -> Dict[str, Any]:
    """Async LangGraph node for the Welcome agent."""
    print("--- Entering Welcome Node ---")
    node_name = "welcome_agent"
    updated_trace = state.agent_trace + [node_name]
    updates: Dict[str, Any] = {"agent_trace": updated_trace}

    try:
        welcome_result = await run_welcome_agent({
            "user_response": state.user_response,
            "conversation_id": state.conversation_id,
            "conversation_history": state.messages
        })
        
        updated_messages = state.messages.copy()
        if welcome_result.message_to_user:
            updated_messages.append({
                "role": "assistant",
                "content": welcome_result.message_to_user
            })
        
        updates.update({
            "welcome_agent_output": welcome_result,
            "messages": updated_messages,
            "current_status": "welcome_complete",
            "quiz_parameters": welcome_result.quiz_parameters  # Add this line
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
        input_data = {
            "user_response": state.user_response,
            "conversation_id": state.conversation_id,
            "conversation_history": state.messages,
            "quiz_parameters": state.quiz_parameters
        }
        print(f"DEBUG: Running Quiz Generator Agent with input: {input_data}")
        
        result = await run_quiz_generator_agent(input_data)
        print(f"DEBUG: Quiz Generator Agent Raw Result: {result}")
        
        # Convert questions to proper format with optional fields
        quiz_output = []
        if result.quiz:
            for q in result.quiz:
                question_dict = q.model_dump() if hasattr(q, 'model_dump') else q
                # Ensure optional fields exist with defaults
                question_dict.setdefault('hint_correct_answer', None)
                question_dict.setdefault('explanation_correct_answer', None)
                question_dict.setdefault('explanation_incorrect_answers', None)
                quiz_output.append(QuizQuestion(**question_dict))

        updates.update({
            "quiz_generator_agent_output": QuizGeneratorAgentOutput(
                message_to_user=result.message_to_user,
                quiz=quiz_output
            ),
            "messages": state.messages + [
                {"role": "assistant", "content": result.message_to_user}
            ],
            "current_status": "quiz_generated"
        })
        
        return updates
        
    except Exception as e:
        print(f"Error in quiz_generator_node: {str(e)}")
        error_output = QuizGeneratorAgentOutput(
            message_to_user="I apologize, but I encountered an error generating your quiz. Could you please try again?",
            quiz=[]
        )
        updates.update({
            "quiz_generator_agent_output": error_output,
            "error_message": str(e),
            "current_status": "error"
        })
        return updates

async def answer_checker_node(state: AppState) -> Dict[str, Any]:
    """Async LangGraph node for the Answer Checker agent."""
    print("--- Entering Answer Checker Node ---")
    node_name = "answer_checker_agent"
    updates: Dict[str, Any] = {"agent_trace": state.agent_trace + [node_name]}
    updated_messages = state.messages.copy()

    # --- Crucial Check: Do we have a quiz and a question index? ---
    if not state.quiz_questions or state.current_question_index < 0 or state.current_question_index >= len(state.quiz_questions):
        print("WARN: Answer Checker Node called without an active question.")
        # Respond gracefully if no question is active
        fallback_message = "It seems there isn't an active question right now. Would you like to start a new quiz?"
        updated_messages.append({"role": "assistant", "content": fallback_message})
        updates.update({
            "messages": updated_messages,
            "current_status": "no_active_question",
            # Provide default output
             "answer_checker_agent_output": AnswerCheckerAgentOutput(message_to_user=fallback_message, is_correct=False, user_answer=state.user_response or "", correct_answer="")
        })
        return updates
    # --- End Check ---

    try:
        # Get context for the current question
        current_question = state.quiz_questions[state.current_question_index]
        question_text = current_question.question
        correct_answer = current_question.correct_answer

        # Run the Answer Checker agent with necessary context
        answer_checker_result = await run_answer_checker_agent({
            "user_response": state.user_response, # The user's answer attempt
            "conversation_id": state.conversation_id,
            "conversation_history": updated_messages,
            "question_text": question_text,
            "correct_answer": correct_answer
        })

        # Add assistant's feedback message
        if answer_checker_result.message_to_user:
            updated_messages.append({"role": "assistant", "content": answer_checker_result.message_to_user})

        # --- State Update after Checking ---
        # Basic: Update output and status.
        # Advanced (Needs Graph Looping): Increment current_question_index, check if quiz ends.
        # For this non-looping graph, we just record the check.
        # A real quiz would need logic here to decide if we ask the *next* question or end.
        next_question_index = state.current_question_index + 1 # Hypothetical next index
        is_quiz_finished = next_question_index >= len(state.quiz_questions)

        updates.update({
            "answer_checker_agent_output": answer_checker_result,
            "messages": updated_messages,
            "current_status": "answer_checked",
            # "current_question_index": next_question_index, # Uncomment if implementing looping
            "error_message": None
        })

        # --- Placeholder for next step logic (if graph looped) ---
        # if not is_quiz_finished:
        #     next_q = state.quiz_questions[next_question_index]
        #     # Add message to ask the next question
        #     ask_next_msg = f"Next question:\n{next_q.question}\nOptions: {', '.join(next_q.options)}"
        #     updated_messages.append({"role": "assistant", "content": ask_next_msg})
        #     updates["messages"] = updated_messages # Update messages again
        # else:
        #     # Add quiz finished message
        #     finish_msg = "You've completed the quiz!"
        #     updated_messages.append({"role": "assistant", "content": finish_msg})
        #     updates["messages"] = updated_messages # Update messages again
        #     updates["current_question_index"] = -1 # Reset index
        # --- End Placeholder ---

        return updates

    except Exception as e:
        error_msg = f"Error in answer_checker_node: {e}"
        print(error_msg)
        updates["error_message"] = error_msg
        fallback_message = "An error occurred while checking your answer."
        updated_messages.append({"role": "assistant", "content": fallback_message})
        updates.update({
            "messages": updated_messages,
            # Provide default output
            "answer_checker_agent_output": AnswerCheckerAgentOutput(message_to_user=fallback_message, is_correct=False, user_answer=state.user_response or "", correct_answer="Unknown")
        })
        return updates

# --- LangGraph Routing Function ---

async def welcome_node_decision(state: AppState) -> str:
    """Simple decision node for welcome agent routing."""
    print("--- Entering Welcome Node Decision ---")
    
    if state.welcome_agent_output:
        agent_to_use = state.welcome_agent_output.agent_to_use
        print(f"DEBUG: Welcome agent routing decision: agent_to_use='{agent_to_use}'")
        
        if agent_to_use == "quiz_generator_agent":
            return "quiz_generator_agent"
        elif agent_to_use == "answer_checker_agent":
            return "answer_checker_agent"
    
    return "end"  # Route to end for any other case

# --- LangGraph Graph Definition ---
workflow = StateGraph(AppState)

# Add nodes
workflow.add_node("welcome_agent", welcome_node)
workflow.add_node("quiz_generator_agent", quiz_generator_node)
workflow.add_node("answer_checker_agent", answer_checker_node)

# Define edges
workflow.add_edge(START, "welcome_agent")

# Conditional routing from Welcome Agent
workflow.add_conditional_edges(
    "welcome_agent",
    welcome_node_decision, # Function to determine the next node
    {
        "quiz_generator_agent": "quiz_generator_agent",
        "answer_checker_agent": "answer_checker_agent",
        "end": END # Explicitly map END state to the graph's END
    }
)

# Edges from specialist agents to END
# NOTE: For a real multi-turn quiz, these would loop back or go to other nodes.


workflow.add_edge("quiz_generator_agent", END)
#workflow.add_edge("answer_checker_agent", END)

# Compile the graph
# No checkpointer added in this refactor for simplicity.
# For persistent conversations/quizzes, add a checkpointer (e.g., SqliteSaver).
try:
    compiled_graph = workflow.compile()
    print("Graph compiled successfully.")
except Exception as e:
    print(f"FATAL: Graph compilation failed: {e}")
    exit()


# --- Main Execution Block ---
from fastapi import FastAPI, Body
from pydantic import BaseModel
import asyncio
import json
import uvicorn

app = FastAPI()

class AppState(BaseModel):
    conversation_id: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    quiz_questions: Optional[List[QuizQuestion]] = None
    current_question_index: Optional[int] = None
    user_response: str

@app.post("/quiz_generator/")
async def quiz_generator_endpoint(state: AppState) -> Dict[str, Any]:
    """FastAPI endpoint for the quiz generator."""
    try:
        # Convert Pydantic model to dict directly without await
        current_state_dict = state.model_dump()
        
        # Run the graph with the state
        final_state = await compiled_graph.ainvoke(current_state_dict)
        
        return {
            "status": "success",
            "state": final_state
        }
        
    except Exception as e:
        print(f"Error in endpoint: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    
    # Run the FastAPI app with uvicorn
    uvicorn.run(
        "quiz-generator.app.agents.quiz_generator:app",  # Import string path to app
        host="127.0.0.1",
        port=8000,
        reload=True,
        workers=1
    )