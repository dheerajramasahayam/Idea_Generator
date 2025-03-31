import sqlite3
import logging
import config # Import configuration for file paths
import os
import datetime
import json # Needed for serializing justifications

DB_FILE = config.STATE_FILE

def init_db(db_path=DB_FILE):
    """Initializes the SQLite database and creates the ideas table if it doesn't exist."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ideas (
                idea_name_lower TEXT PRIMARY KEY,
                original_idea_name TEXT,
                status TEXT NOT NULL,
                rating REAL,
                justifications TEXT, # New column for JSON justifications
                processed_timestamp DATETIME NOT NULL
            )
        ''')
        # Add migration logic here if needed for existing databases
        # For simplicity, we assume a new DB or manual alteration if needed
        # Optional: Add indexes for faster lookups if the table grows large
        # cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON ideas (status);")
        # cursor.execute("CREATE INDEX IF NOT EXISTS idx_rating ON ideas (rating);")
        conn.commit()
        conn.close()
        logging.info(f"Database '{db_path}' initialized successfully.")
    except sqlite3.Error as e:
        logging.error(f"Error initializing database '{db_path}': {e}")
        raise # Re-raise the exception to potentially stop the main script

def load_processed_ideas(db_path=DB_FILE):
    """Loads the set of already processed idea names (lowercase) from the database."""
    processed = set()
    if not os.path.exists(db_path):
        logging.info(f"State database '{db_path}' not found. Starting fresh.")
        return processed

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT idea_name_lower FROM ideas")
        rows = cursor.fetchall()
        processed = {row[0] for row in rows}
        logging.info(f"Loaded {len(processed)} previously processed ideas from '{db_path}'.")
    except sqlite3.Error as e:
        logging.error(f"Error reading processed ideas state from database '{db_path}': {e}")
    finally:
        if conn:
            conn.close()
    return processed

def update_idea_state(idea_name, status, rating=None, justifications=None, db_path=DB_FILE):
    """Inserts or updates the state of an idea in the database, including justifications."""
    conn = None
    idea_name_lower = idea_name.lower()
    timestamp = datetime.datetime.now().isoformat()
    # Serialize justifications dictionary to JSON string, store None if not provided
    justifications_json = json.dumps(justifications) if justifications else None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Use INSERT OR REPLACE to handle both new and existing ideas
        cursor.execute('''
            INSERT OR REPLACE INTO ideas
            (idea_name_lower, original_idea_name, status, rating, justifications, processed_timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (idea_name_lower, idea_name, status, rating, justifications_json, timestamp))
        conn.commit()
        logging.debug(f"Updated state for idea '{idea_name}' to status '{status}' with rating {rating}")
    except sqlite3.Error as e:
        logging.error(f"Error updating state for idea '{idea_name}' in database '{db_path}': {e}")
    finally:
        if conn:
            conn.close()

# --- Functions for potential future use (e.g., negative feedback loop) ---

def get_low_rated_ideas(threshold=4.0, limit=10, db_path=DB_FILE):
    """Retrieves a sample of low-rated ideas from the database."""
    ideas = []
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Select original name for better prompts, order by timestamp descending to get recent ones
        cursor.execute('''
            SELECT original_idea_name FROM ideas
            WHERE status = 'rated' AND rating IS NOT NULL AND rating < ?
            ORDER BY processed_timestamp DESC
            LIMIT ?
        ''', (threshold, limit))
        rows = cursor.fetchall()
        ideas = [row[0] for row in rows]
    except sqlite3.Error as e:
        logging.error(f"Error retrieving low-rated ideas from database '{db_path}': {e}")
    finally:
        if conn:
            conn.close()
    return ideas

def get_high_rated_ideas(threshold=config.RATING_THRESHOLD, limit=50, db_path=DB_FILE):
    """Retrieves high-rated ideas (original name and rating) for trend analysis."""
    ideas_with_ratings = []
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT original_idea_name, rating FROM ideas
            WHERE status = 'rated' AND rating IS NOT NULL AND rating >= ?
            ORDER BY rating DESC
            LIMIT ?
        ''', (threshold, limit))
        rows = cursor.fetchall()
        ideas_with_ratings = [{"name": row[0], "rating": row[1]} for row in rows]
    except sqlite3.Error as e:
        logging.error(f"Error retrieving high-rated ideas from database '{db_path}': {e}")
    finally:
        if conn:
            conn.close()
    return ideas_with_ratings

def get_recent_ratings(limit=50, db_path=DB_FILE):
    """Retrieves the ratings of the most recently processed ideas."""
    ratings = []
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Select non-null ratings, ordered by timestamp descending
        cursor.execute('''
            SELECT rating FROM ideas
            WHERE status = 'rated' AND rating IS NOT NULL
            ORDER BY processed_timestamp DESC
            LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        ratings = [row[0] for row in rows]
    except sqlite3.Error as e:
        logging.error(f"Error retrieving recent ratings from database '{db_path}': {e}")
    finally:
        if conn:
            conn.close()
    return ratings


# --- Keep the function for saving to the final markdown output ---

def save_rated_idea(idea_name, rating, justifications, filename=config.OUTPUT_FILE):
    """Appends a high-scoring idea and its justifications to the main output file."""
    logging.info(f"--- Saving high-scoring idea to '{filename}': '{idea_name}' (Score: {rating:.1f}) ---")
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        # Ensure file exists with header if it's new
        if not os.path.exists(filename):
             with open(filename, 'w', encoding='utf-8') as f:
                  f.write("# AI-Rated SaaS Ideas (Weighted Score >= 7.5)\n\n")
                  f.write("Ideas are rated based on Need, WillingnessToPay, Competition, Monetization, Feasibility.\n")
                  f.write("Justifications are AI-generated based *only* on limited web search summaries.\n\n")

        # Append the new idea and justifications
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(f"## {idea_name} - Score: {rating:.1f}\n")
            f.write(f"- **Need:** {justifications.get('need', 'N/A')}\n")
            f.write(f"- **WillingnessToPay:** {justifications.get('willingnesstopay', 'N/A')}\n")
            f.write(f"- **Competition:** {justifications.get('competition', 'N/A')}\n")
            f.write(f"- **Monetization:** {justifications.get('monetization', 'N/A')}\n")
            f.write(f"- **Feasibility:** {justifications.get('feasibility', 'N/A')}\n\n")
    except Exception as e:
        logging.error(f"Error writing to output file '{filename}': {e}")
