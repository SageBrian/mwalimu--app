from openai import OpenAI
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import instructor
from groq import Groq



load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

client = OpenAI()

def call_llm_api_1(messages: List[Dict[str, str]], 
                model: str = "gpt-4o-mini-2024-07-18",
                response_format: Optional[BaseModel] = None,
                max_tokens: int = 2000,
                temperature: float = 0.3) -> Any:
    """
    Make a call to the OpenAI API for chat completions.
    """
    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format
        )
        return response.choices[0].message.parsed
    except Exception as e:
        print(f"Error in OpenAI API call: {e}")
        raise

    # Groq API


# Patch Groq() with instructor, this is where the magic happens!
groq_client = instructor.from_groq(Groq(api_key=os.getenv("GROQ_API_KEY")), mode=instructor.Mode.JSON)

def call_llm_api(messages: List[Dict[str, str]],
                model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
                response_format: Optional[BaseModel] = None,
                max_tokens: int = 2000,
                temperature: float = 0.3) -> Any:
    """
    Make a call to the Groq API for chat completions.
    """
    try:
        # If a response model is provided, use it for structured output
        if response_format:
            response = groq_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_model=response_format,
                max_retries=3
            )
            # Return the parsed response directly
            return response
        else:
            # For unstructured responses
            response = groq_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                max_retries=3
            )
            return response.choices[0].message.content
    except Exception as e:
        print(f"Error in Groq API call: {e}")
        raise








