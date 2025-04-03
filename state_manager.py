import sqlite3
import logging
import config
import os
import datetime
import json
import random

DB_FILE = config.STATE_FILE

def _does_column_exist(cursor, table_name, column_name):
    """Helper function to check if a column exists in a table."""
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    for col in columns:
        if col[1].lower() == column_name.lower(): return True
    return False

def init_db(db_path=DB_FILE):
    """Initializes the SQLite database and adds missing columns."""
    conn = None
    try:
        logging.info(f"Attempting to initialize database: {db_path}")
        db_dir = os.path.dirname(db_path) or '.'; os.makedirs(db_dir, exist_ok=True)
        logging.debug(f"Ensured directory exists: {db_dir}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        logging.info("Database connection successful. Ensuring 'ideas' table schema...")
        create_table_sql = """
            CREATE TABLE IF NOT EXISTS ideas (
                idea_name_lower TEXT PRIMARY KEY, original_idea_name TEXT, status TEXT NOT NULL,
                rating REAL, processed_timestamp DATETIME NOT NULL ); """
        cursor.execute(create_table_sql)
        columns_to_add = [('justifications', 'TEXT'), ('embedding_json', 'TEXT'), ('description', 'TEXT')]
        for col_name, col_type in columns_to_add:
            if not _does_column_exist(cursor, 'ideas', col_name):
                logging.warning(f"Column '{col_name}' not found. Adding it...")
                cursor.execute(f"ALTER TABLE ideas ADD COLUMN {col_name} {col_type};")
                logging.info(f"Column '{col_name}' added successfully.")
            else: logging.debug(f"Column '{col_name}' already exists.")
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

def update_idea_state(idea_name, status, rating=None, justifications=None, embedding=None, description=None, db_path=DB_FILE):
    """Inserts or updates the state of an idea."""
    conn = None
    idea_name_lower = idea_name.lower()
    timestamp = datetime.datetime.now().isoformat()
    justifications_json = json.dumps(justifications) if justifications else None
    embedding_json = json.dumps(embedding) if embedding else None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO ideas
            (idea_name_lower, original_idea_name, description, status, rating, justifications, embedding_json, processed_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (idea_name_lower, idea_name, description, status, rating, justifications_json, embedding_json, timestamp))
        conn.commit()
        logging.debug(f"Updated state for idea '{idea_name}' to status '{status}'")
    except sqlite3.Error as e: logging.error(f"Error updating state for idea '{idea_name}' in '{db_path}': {e}")
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
    ideas_data = []
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ideas';")
        if cursor.fetchone():
            has_desc = _does_column_exist(cursor, 'ideas', 'description')
            has_embed = _does_column_exist(cursor, 'ideas', 'embedding_json')
            # Select based on rating >= threshold, regardless of status
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
