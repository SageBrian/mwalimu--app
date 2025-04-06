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

logger = setup_logger()


def get_postgres_connection(table_name: str):
    """
    Establish and return a connection to the PostgreSQL database.
    
    :param table_name: Name of the table to interact with
    :return: Connection object
    """
    db_host = os.getenv("DB_HOST", "143.110.249.198").strip()
    db_user = os.getenv("DB_USER", "postgres").strip()
    db_password = os.getenv("DB_PASSWORD", "wes@1234").strip()
    db_port = os.getenv("DB_PORT", "5432").strip()
    db_name = os.getenv("DB_NAME", "postgres").strip()

    try:
        conn = psycopg2.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            port=db_port,
            dbname=db_name
        )
        logger.info(f"Successfully connected to database: {db_name}")
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"Unable to connect to the database. Error: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise

