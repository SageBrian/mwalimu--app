from .db import get_postgres_connection
from .logger_setup import setup_logger
from psycopg2.extras import Json

logger = setup_logger()

def save_conversation(result: dict):
    """Save conversation result to database"""
    conn = get_postgres_connection("conversations")
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO uliza_conversations 
                (log_timestamp, user_id, session_id, conversation_id, user_input, state)
                VALUES
                (NOW(), %s, %s, %s, %s, %s)
                RETURNING id;
            """, (
                result.get("user_id"),
                result.get("session_id"),
                result.get("conversation_id"),
                result.get("user_input"),
                Json(result)
            ))
            inserted_id = cur.fetchone()[0]
        conn.commit()
        logger.info(f"Conversation saved with ID: {inserted_id}")
        return inserted_id
    except Exception as e:
        logger.error(f"Error saving conversation: {e}")
        raise
    finally:
        conn.close()
