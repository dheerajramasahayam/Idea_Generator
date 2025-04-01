import sqlite3
import logging
import config # Import configuration for file paths
import os
import datetime
import json # Needed for serializing justifications and embeddings

DB_FILE = config.STATE_FILE

def _does_column_exist(cursor, table_name, column_name):
    """Helper function to check if a column exists in a table."""
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    for col in columns:
        if col[1].lower() == column_name.lower():
            return True
    return False

def init_db(db_path=DB_FILE):
    """
    Initializes the SQLite database. Creates the table if it doesn't exist,
    and adds missing columns ('justifications', 'embedding_json').
    """
    conn = None
    try:
        logging.info(f"Attempting to initialize database: {db_path}")
        db_dir = os.path.dirname(db_path) or '.'
        os.makedirs(db_dir, exist_ok=True)
        logging.debug(f"Ensured directory exists: {db_dir}")

        logging.debug(f"Connecting to database: {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        logging.info("Database connection successful. Ensuring 'ideas' table schema...")

        # Step 1: Ensure table exists (basic structure)
        create_table_sql = """
            CREATE TABLE IF NOT EXISTS ideas (
                idea_name_lower TEXT PRIMARY KEY,
                original_idea_name TEXT,
                status TEXT NOT NULL,
                rating REAL,
                processed_timestamp DATETIME NOT NULL
            );
        """
        logging.debug(f"Executing: {create_table_sql.strip()}")
        cursor.execute(create_table_sql)
        logging.debug("CREATE TABLE IF NOT EXISTS executed.")

        # Step 2: Check/Add 'justifications' column
        if not _does_column_exist(cursor, 'ideas', 'justifications'):
            logging.warning("Column 'justifications' not found. Adding it...")
            cursor.execute("ALTER TABLE ideas ADD COLUMN justifications TEXT;")
            logging.info("Column 'justifications' added successfully.")
        else:
            logging.debug("Column 'justifications' already exists.")

        # Step 3: Check/Add 'embedding_json' column
        if not _does_column_exist(cursor, 'ideas', 'embedding_json'):
            logging.warning("Column 'embedding_json' not found. Adding it...")
            cursor.execute("ALTER TABLE ideas ADD COLUMN embedding_json TEXT;")
            logging.info("Column 'embedding_json' added successfully.")
        else:
            logging.debug("Column 'embedding_json' already exists.")

        conn.commit()
        logging.info(f"Database '{db_path}' schema verified/updated successfully.")

    except sqlite3.Error as e:
        logging.error(f"SQLite error during database initialization '{db_path}': {e}")
        raise
    except Exception as e:
        logging.error(f"Non-SQLite error during database initialization '{db_path}': {e}")
        raise
    finally:
        if conn:
            conn.close()
            logging.debug(f"Database connection closed for '{db_path}'.")


def load_processed_ideas(db_path=DB_FILE):
    """Loads the set of already processed idea names (lowercase) from the database."""
    processed = set()
    if not os.path.exists(db_path): return processed
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ideas';")
        if cursor.fetchone():
            cursor.execute("SELECT idea_name_lower FROM ideas")
            rows = cursor.fetchall()
            processed = {row[0] for row in rows}
            logging.info(f"Loaded {len(processed)} previously processed ideas from '{db_path}'.")
        else: logging.warning(f"Table 'ideas' not found in '{db_path}'.")
    except sqlite3.Error as e: logging.error(f"Error reading state from '{db_path}': {e}")
    finally:
        if conn: conn.close()
    return processed

def update_idea_state(idea_name, status, rating=None, justifications=None, embedding=None, db_path=DB_FILE):
    """Inserts or updates the state of an idea, including justifications and embedding."""
    conn = None
    idea_name_lower = idea_name.lower()
    timestamp = datetime.datetime.now().isoformat()
    justifications_json = json.dumps(justifications) if justifications else None
    embedding_json = json.dumps(embedding) if embedding else None # Serialize embedding list to JSON
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO ideas
            (idea_name_lower, original_idea_name, status, rating, justifications, embedding_json, processed_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (idea_name_lower, idea_name, status, rating, justifications_json, embedding_json, timestamp))
        conn.commit()
        logging.debug(f"Updated state for idea '{idea_name}' to status '{status}'")
    except sqlite3.Error as e:
        logging.error(f"Error updating state for idea '{idea_name}' in '{db_path}': {e}")
    finally:
        if conn: conn.close()

def get_low_rated_embeddings(threshold=4.0, limit=100, db_path=DB_FILE):
    """Retrieves embeddings (as lists) of low-rated ideas."""
    embeddings = []
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ideas';")
        if cursor.fetchone():
            # Check if embedding column exists before querying it
            if _does_column_exist(cursor, 'ideas', 'embedding_json'):
                cursor.execute('''
                    SELECT embedding_json FROM ideas
                    WHERE status IN ('rated', 'saved', 'error', 'rating_failed') -- Consider all processed low-rated
                    AND rating IS NOT NULL AND rating < ?
                    AND embedding_json IS NOT NULL -- Only get those with embeddings
                    ORDER BY processed_timestamp DESC
                    LIMIT ?
                ''', (threshold, limit))
                rows = cursor.fetchall()
                for row in rows:
                    try:
                        # Deserialize JSON string back to list
                        embedding_list = json.loads(row[0])
                        if isinstance(embedding_list, list):
                            embeddings.append(embedding_list)
                    except (json.JSONDecodeError, TypeError):
                        logging.warning(f"Could not decode embedding JSON for a low-rated idea: {row[0][:50]}...")
            else:
                 logging.warning("Cannot get low-rated embeddings: 'embedding_json' column not found.")
        else:
            logging.warning("Cannot get low-rated embeddings: 'ideas' table not found.")
    except sqlite3.Error as e:
        logging.error(f"Error retrieving low-rated embeddings from '{db_path}': {e}")
    finally:
        if conn: conn.close()
    logging.info(f"Retrieved {len(embeddings)} embeddings for low-rated ideas.")
    return embeddings


def get_high_rated_ideas(threshold=config.RATING_THRESHOLD, limit=50, db_path=DB_FILE):
    """Retrieves high-rated ideas (name and rating) for trend analysis."""
    ideas_with_ratings = []
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ideas';")
        if cursor.fetchone():
            cursor.execute('''
                SELECT original_idea_name, rating FROM ideas
                WHERE status = 'saved' -- Focus trends on ideas that met the final threshold
                AND rating IS NOT NULL AND rating >= ?
                ORDER BY rating DESC, processed_timestamp DESC
                LIMIT ?
            ''', (threshold, limit))
            rows = cursor.fetchall()
            ideas_with_ratings = [{"name": row[0], "rating": row[1]} for row in rows]
        else: logging.warning("Cannot get high-rated ideas: 'ideas' table not found.")
    except sqlite3.Error as e: logging.error(f"Error retrieving high-rated ideas from '{db_path}': {e}")
    finally:
        if conn: conn.close()
    return ideas_with_ratings

def get_recent_ratings(limit=50, db_path=DB_FILE):
    """Retrieves the ratings of the most recently processed ideas."""
    ratings = []
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ideas';")
        if cursor.fetchone():
            cursor.execute('''
                SELECT rating FROM ideas
                WHERE status IN ('rated', 'saved') AND rating IS NOT NULL
                ORDER BY processed_timestamp DESC
                LIMIT ?
            ''', (limit,))
            rows = cursor.fetchall()
            ratings = [row[0] for row in rows]
        else: logging.warning("Cannot get recent ratings: 'ideas' table not found.")
    except sqlite3.Error as e: logging.error(f"Error retrieving recent ratings from '{db_path}': {e}")
    finally:
        if conn: conn.close()
    return ratings

# --- Keep the function for saving to the final markdown output ---
def save_rated_idea(idea_name, rating, justifications, filename=config.OUTPUT_FILE):
    """Appends a high-scoring idea and its justifications to the main output file."""
    logging.info(f"--- Saving high-scoring idea to '{filename}': '{idea_name}' (Score: {rating:.1f}) ---")
    try:
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        if not os.path.exists(filename):
             with open(filename, 'w', encoding='utf-8') as f:
                  f.write(f"# AI-Rated SaaS Ideas (Weighted Score >= {config.RATING_THRESHOLD})\n\n[...Header...]\n\n")
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(f"## {idea_name} - Score: {rating:.1f}\n")
            f.write(f"- **Need:** {justifications.get('need', 'N/A')}\n")
            f.write(f"- **WillingnessToPay:** {justifications.get('willingnesstopay', 'N/A')}\n")
            f.write(f"- **Competition:** {justifications.get('competition', 'N/A')}\n")
            f.write(f"- **Monetization:** {justifications.get('monetization', 'N/A')}\n")
            f.write(f"- **Feasibility:** {justifications.get('feasibility', 'N/A')}\n\n")
    except Exception as e:
        logging.error(f"Error writing to output file '{filename}': {e}")
