from .db import get_postgres_connection
from .logger_setup import setup_logger
from psycopg2.extras import Json
from datetime import datetime, timezone

logger = setup_logger()

def save_conversation(result: dict):
    """Save conversation result to database"""
    conn = get_postgres_connection("conversations")
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO mwalimu_conversations 
                (log_timestamp, user_id, session_id, phone_number, user_input, state, created_at)
                VALUES
                (NOW(), %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (phone_number) 
                DO UPDATE SET
                    log_timestamp = NOW(),
                    user_id = EXCLUDED.user_id,
                    session_id = EXCLUDED.session_id,
                    user_input = EXCLUDED.user_input,
                    state = EXCLUDED.state
                RETURNING id;
            """, (
                result.get("user_id"),
                result.get("session_id", None),  # Make session_id optional
                result.get("phone_number"),
                result.get("user_input"),
                Json(result)
            ))
            inserted_id = cur.fetchone()[0]
        conn.commit()
        logger.info(f"Conversation upserted with ID: {inserted_id}")
        return inserted_id
    except Exception as e:
        logger.error(f"Error saving conversation: {e}")
        raise
    finally:
        conn.close()

def load_conversation(phone_number: str):
    """Load conversation from database"""
    conn = get_postgres_connection("conversations")
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT state FROM mwalimu_conversations 
                WHERE phone_number = %s
                ORDER BY log_timestamp DESC 
                LIMIT 1
            """, (phone_number,))
            result = cur.fetchone()
            return result[0] if result else None
    except Exception as e:
        logger.error(f"Error loading conversation: {e}")
        raise
    finally:
        conn.close()

