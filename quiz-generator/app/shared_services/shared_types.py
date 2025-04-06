   # shared_types.py
from typing import TypedDict

class MainState(TypedDict):
    user_id: str
    session_id: str
    conversation_id: str
    user_input: str
    conversation_history: list

