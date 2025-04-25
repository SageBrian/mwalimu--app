import os

from typing import List, Dict, Any, Optional, TypedDict, Union
from dotenv import load_dotenv
import google.generativeai as genai
import psycopg2
import requests
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolExecutor
from psycopg2.extras import Json, RealDictCursor
from google.generativeai import GenerativeModel
import numpy as np
from openai import OpenAI
from tavily import TavilyClient
from .logger_setup import setup_logger

# Load environment variables
load_dotenv()

logger = setup_logger()


def get_postgres_connection(table_name: str):
    """
    Establish and return a connection to the PostgreSQL database using Supabase direct connection.
    For direct database connections, we use the connection string from Supabase Dashboard:
    Settings -> Database -> Connection string -> URI
    
    :param table_name: Name of the table to interact with
    :return: Connection object
    """
    # Get Supabase connection details from environment variables
    db_host = os.getenv("PGHOST")  # Host from connection string
    db_password = os.getenv("PGPASSWORD")  # Password from connection string
    db_port = os.getenv("5432")
    db_name = os.getenv("PGDATABASE")
    db_user = os.getenv("PGUSER")  # User from connection string



    if not all([db_host, db_password, db_user]):
        error_msg = "Missing required  database credentials in environment variables"
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        # Direct PostgreSQL connection to Supabase database
        conn = psycopg2.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_password,
            port=db_port
        
        )
        logger.info(f"Successfully connected to Supabase database: {db_name}")
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"Unable to connect to  database. Error: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while connecting to Supabase: {e}")
        raise

