from db import get_postgres_connection

def create_conversations_table():
    """Create conversations table if it doesn't exist"""
    conn = get_postgres_connection("conversations")
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS uliza_conversations (
                    id SERIAL PRIMARY KEY,
                    log_timestamp TIMESTAMP WITH TIME ZONE,
                    user_id VARCHAR(255),
                    session_id VARCHAR(255),
                    conversation_id VARCHAR(255),
                    user_input TEXT,
                    state JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
        conn.commit()
        print("Conversations table created or already exists")
    except Exception as e:
        print(f"Error creating table: {e}")
        raise
    finally:
        conn.close()
