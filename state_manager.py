import sqlite3
import logging
import config
import os
import datetime
import json
import random
import math # For softmax calculation

DB_FILE = config.STATE_FILE

def _does_column_exist(cursor, table_name, column_name):
    """Helper function to check if a column exists in a table."""
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    for col in columns:
        if col[1].lower() == column_name.lower(): return True
    return False

def init_db(db_path=DB_FILE):
    """Initializes the SQLite database and adds missing columns/tables."""
    conn = None
    try:
        logging.info(f"Attempting to initialize database: {db_path}")
        db_dir = os.path.dirname(db_path) or '.'; os.makedirs(db_dir, exist_ok=True)
        logging.debug(f"Ensured directory exists: {db_dir}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        logging.info("Database connection successful. Ensuring 'ideas' table schema...")
        # --- Ideas Table ---
        create_ideas_table_sql = """
            CREATE TABLE IF NOT EXISTS ideas (
                idea_name_lower TEXT PRIMARY KEY, original_idea_name TEXT, status TEXT NOT NULL,
                rating REAL, processed_timestamp DATETIME NOT NULL ); """
        cursor.execute(create_ideas_table_sql)
        columns_to_add = [
            ('justifications', 'TEXT'), ('embedding_json', 'TEXT'),
            ('description', 'TEXT'), ('source_prompt_type', 'TEXT') # Added source_prompt_type
        ]
        for col_name, col_type in columns_to_add:
            if not _does_column_exist(cursor, 'ideas', col_name):
                logging.warning(f"Column '{col_name}' not found in 'ideas'. Adding it...")
                cursor.execute(f"ALTER TABLE ideas ADD COLUMN {col_name} {col_type};")
                logging.info(f"Column '{col_name}' added successfully.")
            else: logging.debug(f"Column '{col_name}' already exists in 'ideas'.")

        # --- Pending Ideas Table ---
        logging.info("Ensuring 'pending_ideas' table schema...")
        create_pending_table_sql = """
            CREATE TABLE IF NOT EXISTS pending_ideas (
                idea_name TEXT PRIMARY KEY, added_timestamp DATETIME NOT NULL ); """
        cursor.execute(create_pending_table_sql)

        # --- Prompt Performance Table ---
        logging.info("Ensuring 'prompt_performance' table schema...")
        create_perf_table_sql = """
            CREATE TABLE IF NOT EXISTS prompt_performance (
                prompt_type TEXT PRIMARY KEY,
                total_generated INTEGER DEFAULT 0 NOT NULL,
                total_rating REAL DEFAULT 0.0 NOT NULL,
                average_rating REAL DEFAULT 0.0 NOT NULL
            );
        """
        cursor.execute(create_perf_table_sql)

        conn.commit()
        logging.info(f"Database '{db_path}' schema verified/updated successfully.")
    except sqlite3.Error as e: logging.error(f"SQLite error during DB init '{db_path}': {e}"); raise
    except Exception as e: logging.error(f"Non-SQLite error during DB init '{db_path}': {e}"); raise
    finally:
        if conn: conn.close(); logging.debug(f"DB connection closed for '{db_path}'.")

def load_processed_ideas(db_path=DB_FILE):
    """Loads the set of already processed idea names (lowercase)."""
    processed = set()
    if not os.path.exists(db_path): return processed
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ideas';")
        if cursor.fetchone():
            cursor.execute("SELECT idea_name_lower FROM ideas")
            processed = {row[0] for row in cursor.fetchall()}
            logging.info(f"Loaded {len(processed)} previously processed ideas from '{db_path}'.")
        else: logging.warning(f"Table 'ideas' not found in '{db_path}'.")
    except sqlite3.Error as e: logging.error(f"Error reading state from '{db_path}': {e}")
    finally:
        if conn: conn.close()
    return processed

def update_idea_state(idea_name, status, rating=None, justifications=None, embedding=None, description=None, source_prompt_type=None, db_path=DB_FILE):
    """Inserts or updates the state of an idea, including its source prompt type."""
    conn = None
    idea_name_lower = idea_name.lower()
    timestamp = datetime.datetime.now().isoformat()
    justifications_json = json.dumps(justifications) if justifications else None
    embedding_json = json.dumps(embedding) if embedding else None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Ensure source_prompt_type column exists before trying to update it
        if not _does_column_exist(cursor, 'ideas', 'source_prompt_type'):
             source_prompt_type = None # Set to None if column doesn't exist

        cursor.execute('''
            INSERT OR REPLACE INTO ideas
            (idea_name_lower, original_idea_name, description, status, rating, justifications, embedding_json, source_prompt_type, processed_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (idea_name_lower, idea_name, description, status, rating, justifications_json, embedding_json, source_prompt_type, timestamp))
        conn.commit()
        logging.debug(f"Updated state for idea '{idea_name}' to status '{status}' (Source: {source_prompt_type})")
    except sqlite3.Error as e: logging.error(f"Error updating state for idea '{idea_name}' in '{db_path}': {e}")
    finally:
        if conn: conn.close()

# --- Functions for Pending Ideas Queue ---

def add_pending_ideas(idea_names, db_path=DB_FILE):
    """Adds a list of idea names to the pending queue."""
    if not idea_names: return
    conn = None
    timestamp = datetime.datetime.now().isoformat()
    ideas_to_insert = [(name, timestamp) for name in idea_names]
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.executemany('''
            INSERT OR IGNORE INTO pending_ideas (idea_name, added_timestamp) VALUES (?, ?)
        ''', ideas_to_insert)
        conn.commit()
        logging.info(f"Added {len(ideas_to_insert)} ideas to the pending queue.")
    except sqlite3.Error as e: logging.error(f"Error adding pending ideas to '{db_path}': {e}")
    finally:
        if conn: conn.close()

def fetch_and_clear_pending_ideas(limit, db_path=DB_FILE):
    """Fetches a batch of pending ideas and removes them from the queue."""
    pending_ideas = []
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT idea_name FROM pending_ideas ORDER BY added_timestamp ASC LIMIT ?", (limit,))
        pending_ideas = [row[0] for row in cursor.fetchall()]
        if pending_ideas:
            placeholders = ','.join('?' for _ in pending_ideas)
            cursor.execute(f"DELETE FROM pending_ideas WHERE idea_name IN ({placeholders})", pending_ideas)
            conn.commit()
            logging.info(f"Fetched and cleared {len(pending_ideas)} ideas from the pending queue.")
        else: logging.debug("No pending ideas found in the queue.")
    except sqlite3.Error as e: logging.error(f"Error fetching/clearing pending ideas from '{db_path}': {e}")
    finally:
        if conn: conn.close()
    return pending_ideas

# --- Functions for Prompt Performance Tracking ---

def update_prompt_performance(prompt_type, rating, db_path=DB_FILE):
    """Updates the performance metrics for a given prompt type."""
    if rating is None or prompt_type is None: return # Cannot update without rating and type
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Ensure row exists
        cursor.execute("INSERT OR IGNORE INTO prompt_performance (prompt_type) VALUES (?)", (prompt_type,))
        # Update totals
        cursor.execute("""
            UPDATE prompt_performance
            SET total_generated = total_generated + 1,
                total_rating = total_rating + ?
            WHERE prompt_type = ?
        """, (rating, prompt_type))
        # Recalculate and update average
        cursor.execute("""
            UPDATE prompt_performance
            SET average_rating = total_rating / total_generated
            WHERE prompt_type = ? AND total_generated > 0
        """, (prompt_type,))
        conn.commit()
        logging.debug(f"Updated performance for prompt type '{prompt_type}' with rating {rating:.1f}")
    except sqlite3.Error as e: logging.error(f"Error updating prompt performance for '{prompt_type}': {e}")
    finally:
        if conn: conn.close()

def get_prompt_performance(db_path=DB_FILE):
    """Retrieves performance data for all tracked prompt types."""
    performance_data = {}
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT prompt_type, total_generated, average_rating FROM prompt_performance")
        for row in cursor.fetchall():
            prompt_type, count, avg_rating = row
            performance_data[prompt_type] = {'count': count, 'avg_rating': avg_rating}
        logging.info(f"Retrieved performance data for {len(performance_data)} prompt types.")
    except sqlite3.Error as e: logging.error(f"Error retrieving prompt performance data: {e}")
    finally:
        if conn: conn.close()
    return performance_data


# --- Functions for Analysis/Feedback ---

def get_low_rated_embeddings(threshold=4.0, limit=100, db_path=DB_FILE):
    """Retrieves embeddings (as lists) of low-rated ideas."""
    # ... (implementation remains the same) ...
    embeddings = []
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ideas';")
        if cursor.fetchone() and _does_column_exist(cursor, 'ideas', 'embedding_json'):
            cursor.execute(''' SELECT embedding_json FROM ideas WHERE status IN ('rated', 'saved', 'error', 'rating_failed')
                                AND rating IS NOT NULL AND rating < ? AND embedding_json IS NOT NULL
                                ORDER BY processed_timestamp DESC LIMIT ? ''', (threshold, limit))
            for row in cursor.fetchall():
                try:
                    emb = json.loads(row[0]);
                    if isinstance(emb, list): embeddings.append(emb)
                except (json.JSONDecodeError, TypeError): logging.warning(f"Could not decode embedding JSON: {row[0][:50]}...")
        else: logging.warning("Cannot get low-rated embeddings: table or column missing.")
    except sqlite3.Error as e: logging.error(f"Error retrieving low-rated embeddings from '{db_path}': {e}")
    finally:
        if conn: conn.close()
    logging.info(f"Retrieved {len(embeddings)} embeddings for low-rated ideas.")
    return embeddings

def get_low_rated_texts(threshold=4.0, limit=50, db_path=DB_FILE):
    """Retrieves names and descriptions of low-rated ideas for keyword extraction."""
    # ... (implementation remains the same) ...
    texts = []
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ideas';")
        if cursor.fetchone():
            has_desc = _does_column_exist(cursor, 'ideas', 'description')
            select_cols = "original_idea_name" + (", description" if has_desc else "")
            cursor.execute(f''' SELECT {select_cols} FROM ideas WHERE status IN ('rated', 'saved', 'error', 'rating_failed')
                                AND rating IS NOT NULL AND rating < ? ORDER BY processed_timestamp DESC LIMIT ? ''', (threshold, limit))
            for row in cursor.fetchall():
                texts.append(row[0]) # Add name
                if has_desc and row[1]: texts.append(row[1]) # Add description if exists
        else: logging.warning("Cannot get low-rated texts: 'ideas' table not found.")
    except sqlite3.Error as e: logging.error(f"Error retrieving low-rated texts from '{db_path}': {e}")
    finally:
        if conn: conn.close()
    logging.info(f"Retrieved {len(texts)} texts (names/descriptions) from low-rated ideas for keyword analysis.")
    return texts


def get_high_rated_ideas(threshold=config.RATING_THRESHOLD, limit=50, db_path=DB_FILE):
    """Retrieves high-rated ideas (name, desc, rating, embedding) for trend analysis, based purely on rating threshold."""
    # ... (implementation remains the same - already filters only on rating) ...
    ideas_data = []
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ideas';")
        if cursor.fetchone():
            has_desc = _does_column_exist(cursor, 'ideas', 'description')
            has_embed = _does_column_exist(cursor, 'ideas', 'embedding_json')
            select_cols = "original_idea_name, rating" + (", description" if has_desc else "") + (", embedding_json" if has_embed else "")
            cursor.execute(f''' SELECT {select_cols} FROM ideas
                                WHERE rating IS NOT NULL AND rating >= ?
                                ORDER BY rating DESC, processed_timestamp DESC LIMIT ? ''', (threshold, limit))
            rows = cursor.fetchall()
            logging.info(f"Found {len(rows)} ideas with rating >= {threshold} for trend analysis.")
            for row in rows:
                 data = {"name": row[0], "rating": row[1]}; col_index = 2
                 if has_desc: data["description"] = row[col_index]; col_index += 1
                 if has_embed:
                      try: data["embedding"] = json.loads(row[col_index]) if row[col_index] else None
                      except (json.JSONDecodeError, TypeError): data["embedding"] = None
                 ideas_data.append(data)
        else: logging.warning("Cannot get high-rated ideas: 'ideas' table not found.")
    except sqlite3.Error as e: logging.error(f"Error retrieving high-rated ideas from '{db_path}': {e}")
    finally:
        if conn: conn.close()
    return ideas_data

def get_variation_candidate_ideas(min_rating, max_rating, limit=10, db_path=DB_FILE):
    """Retrieves moderately rated ideas suitable for generating variations."""
    # ... (implementation remains the same) ...
    candidates = []
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ideas';")
        if cursor.fetchone():
            if not _does_column_exist(cursor, 'ideas', 'description'): logging.warning("Variation candidates: 'description' column missing."); return []
            if not _does_column_exist(cursor, 'ideas', 'justifications'): logging.warning("Variation candidates: 'justifications' column missing."); return []
            cursor.execute(''' SELECT original_idea_name, description, rating, justifications FROM ideas
                                WHERE status IN ('rated', 'saved') AND rating IS NOT NULL AND rating >= ? AND rating <= ?
                                AND description IS NOT NULL AND justifications IS NOT NULL ORDER BY RANDOM() LIMIT ? ''',
                           (min_rating, max_rating, limit))
            for row in cursor.fetchall():
                try:
                    justifications_dict = json.loads(row[3])
                    if isinstance(justifications_dict, dict):
                        candidates.append({"name": row[0], "description": row[1], "rating": row[2], "justifications": justifications_dict})
                except (json.JSONDecodeError, TypeError): logging.warning(f"Could not decode justifications for candidate '{row[0]}'")
        else: logging.warning("Cannot get variation candidates: 'ideas' table not found.")
    except sqlite3.Error as e: logging.error(f"Error retrieving variation candidates from '{db_path}': {e}")
    finally:
        if conn: conn.close()
    logging.info(f"Retrieved {len(candidates)} potential candidates for variation generation.")
    return candidates

def get_recent_ratings(limit=50, db_path=DB_FILE):
    """Retrieves the ratings of the most recently processed ideas."""
    # ... (implementation remains the same) ...
    ratings = []
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ideas';")
        if cursor.fetchone():
            cursor.execute(''' SELECT rating FROM ideas WHERE status IN ('rated', 'saved') AND rating IS NOT NULL
                                ORDER BY processed_timestamp DESC LIMIT ? ''', (limit,))
            ratings = [row[0] for row in cursor.fetchall()]
        else: logging.warning("Cannot get recent ratings: 'ideas' table not found.")
    except sqlite3.Error as e: logging.error(f"Error retrieving recent ratings from '{db_path}': {e}")
    finally:
        if conn: conn.close()
    return ratings

def save_rated_idea(idea_name, rating, justifications, filename=config.OUTPUT_FILE):
    """Appends a high-scoring idea and its justifications to the main output file."""
    # ... (implementation remains the same) ...
    logging.info(f"--- Saving high-scoring idea to '{filename}': '{idea_name}' (Score: {rating:.1f}) ---")
    try:
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        if not os.path.exists(filename):
             with open(filename, 'w', encoding='utf-8') as f: f.write(f"# AI-Rated SaaS Ideas (Score >= {config.RATING_THRESHOLD})\n\n[...Header...]\n\n")
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(f"## {idea_name} - Score: {rating:.1f}\n")
            f.write(f"- **Need:** {justifications.get('need', 'N/A')}\n")
            f.write(f"- **WillingnessToPay:** {justifications.get('willingnesstopay', 'N/A')}\n")
            f.write(f"- **Competition:** {justifications.get('competition', 'N/A')}\n")
            f.write(f"- **Monetization:** {justifications.get('monetization', 'N/A')}\n")
            f.write(f"- **Feasibility:** {justifications.get('feasibility', 'N/A')}\n\n")
    except Exception as e: logging.error(f"Error writing to output file '{filename}': {e}")
