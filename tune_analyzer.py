# tune_analyzer.py
import sqlite3
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
import sys

# Add project root to path to allow importing project modules
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import config # To get DB path, thresholds etc.
    import state_manager
    import analysis_utils
    from sentence_transformers import util # For similarity calculation
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Ensure this script is run from the project root directory or the path is configured correctly.")
    sys.exit(1)
except FileNotFoundError:
     print("Error: config.py or other required project files not found.")
     print("Ensure this script is run from the project root directory.")
     sys.exit(1)


# --- Configuration ---
DB_FILE = config.STATE_FILE
OUTPUT_DIR = "analysis_output"
SIMILARITY_SAMPLE_SIZE = 1000 # Limit pairwise comparisons for performance
TREND_THRESHOLDS_TO_TEST = [7.5, 8.0, 8.5, 9.0] # Test different thresholds

# --- Logging Setup ---
# Use config's setup if available, otherwise basic config
try:
    config.setup_logging()
    logging.getLogger().setLevel(logging.INFO) # Ensure level is INFO
except AttributeError:
     print("Using basic logging config.")
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Helper Functions ---

def load_all_idea_data(db_path=DB_FILE):
    """Loads relevant data (name, desc, rating, embedding) for all processed ideas."""
    data = []
    conn = None
    logging.info(f"Loading data from database: {db_path}")
    if not os.path.exists(db_path):
        logging.error(f"Database file not found: {db_path}")
        return data

    required_cols = ['original_idea_name', 'description', 'rating', 'embedding_json', 'status', 'justifications']

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ideas';")
        if not cursor.fetchone():
            logging.error("Table 'ideas' not found in database.")
            return data

        # Check for required columns
        missing_cols = [col for col in required_cols if not state_manager._does_column_exist(cursor, 'ideas', col)]
        if missing_cols:
            # Allow missing embedding/description/justifications for backward compatibility if needed
            logging.warning(f"Database might be missing some columns used by analysis: {', '.join(missing_cols)}")
            # Adjust required cols based on what's essential for basic analysis vs full
            required_cols = ['original_idea_name', 'rating', 'status'] # Minimum needed
            missing_essential = [col for col in required_cols if not state_manager._does_column_exist(cursor, 'ideas', col)]
            if missing_essential:
                 logging.error(f"Database missing essential columns for analysis: {', '.join(missing_essential)}")
                 return data

        # Construct SELECT query based on existing columns
        available_cols = ['original_idea_name', 'rating', 'status']
        if state_manager._does_column_exist(cursor, 'ideas', 'description'): available_cols.append('description')
        if state_manager._does_column_exist(cursor, 'ideas', 'embedding_json'): available_cols.append('embedding_json')
        if state_manager._does_column_exist(cursor, 'ideas', 'justifications'): available_cols.append('justifications')

        select_str = ", ".join(available_cols)
        cursor.execute(f"SELECT {select_str} FROM ideas WHERE status IN ('rated', 'saved')") # Analyze rated & saved
        rows = cursor.fetchall()
        logging.info(f"Loaded {len(rows)} processed ideas from DB.")

        col_names = [desc[0] for desc in cursor.description] # Get column names from cursor

        for row in rows:
            item = dict(zip(col_names, row)) # Create dict from row
            # Process embedding
            embedding = None
            embedding_json = item.get('embedding_json')
            if embedding_json:
                try:
                    embedding = json.loads(embedding_json)
                    if not isinstance(embedding, list): embedding = None
                except (json.JSONDecodeError, TypeError): embedding = None
            item['embedding'] = embedding # Add processed embedding

            # Process justifications
            justifications = None
            justifications_json = item.get('justifications')
            if justifications_json:
                 try:
                      justifications = json.loads(justifications_json)
                      if not isinstance(justifications, dict): justifications = None
                 except (json.JSONDecodeError, TypeError): justifications = None
            item['justifications'] = justifications # Add processed justifications

            data.append(item)

    except sqlite3.Error as e:
        logging.error(f"SQLite error loading data: {e}")
    except Exception as e:
        logging.error(f"Unexpected error loading data: {e}")
    finally:
        if conn: conn.close()
    return data

def analyze_ratings(all_data):
    """Calculates and prints rating statistics and plots distribution."""
    ratings = [item['rating'] for item in all_data if item.get('rating') is not None]
    if not ratings:
        logging.warning("No rating data found to analyze.")
        return

    ratings_np = np.array(ratings)
    logging.info("\n--- Rating Analysis ---")
    logging.info(f"Total ideas with ratings: {len(ratings_np)}")
    logging.info(f"Mean Rating: {np.mean(ratings_np):.2f}")
    logging.info(f"Median Rating: {np.median(ratings_np):.2f}")
    logging.info(f"Std Dev Rating: {np.std(ratings_np):.2f}")
    logging.info(f"Min Rating: {np.min(ratings_np):.2f}")
    logging.info(f"Max Rating: {np.max(ratings_np):.2f}")
    logging.info(f"25th Percentile: {np.percentile(ratings_np, 25):.2f}")
    logging.info(f"75th Percentile: {np.percentile(ratings_np, 75):.2f}")
    logging.info(f"Ideas >= 9.0: {np.sum(ratings_np >= 9.0)}")
    logging.info(f"Ideas >= 8.5: {np.sum(ratings_np >= 8.5)}")
    logging.info(f"Ideas >= 8.0: {np.sum(ratings_np >= 8.0)}")
    logging.info(f"Ideas >= 7.5: {np.sum(ratings_np >= 7.5)}")


    # Plot Histogram
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(ratings_np, bins=20, range=(0, 10), edgecolor='black')
        plt.title('Distribution of Idea Ratings')
        plt.xlabel('Rating Score')
        plt.ylabel('Number of Ideas')
        plt.xticks(np.arange(0, 10.5, 0.5))
        plt.grid(axis='y', alpha=0.75)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plot_path = os.path.join(OUTPUT_DIR, "rating_distribution.png")
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Rating distribution histogram saved to: {plot_path}")
    except Exception as e:
        logging.error(f"Failed to generate rating histogram: {e}")

def analyze_similarity(all_data):
    """Calculates and plots pairwise semantic similarity of descriptions."""
    embeddings = [item['embedding'] for item in all_data if item.get('embedding') is not None]
    if len(embeddings) < 2:
        logging.warning("Not enough embeddings found (<2) to analyze similarity.")
        return

    logging.info("\n--- Semantic Similarity Analysis (Description Embeddings) ---")
    logging.info(f"Analyzing {len(embeddings)} embeddings.")

    # Sample if too large
    if len(embeddings) > SIMILARITY_SAMPLE_SIZE:
        logging.warning(f"Sampling {SIMILARITY_SAMPLE_SIZE} embeddings for similarity analysis.")
        indices = np.random.choice(len(embeddings), SIMILARITY_SAMPLE_SIZE, replace=False)
        embeddings_sample = [embeddings[i] for i in indices]
    else:
        embeddings_sample = embeddings

    embeddings_np = np.array(embeddings_sample)

    try:
        logging.info("Calculating pairwise cosine similarities (this may take a while)...")
        cosine_scores = util.cos_sim(embeddings_np, embeddings_np).numpy()
        upper_triangle_indices = np.triu_indices_from(cosine_scores, k=1)
        similarity_values = cosine_scores[upper_triangle_indices]

        if similarity_values.size == 0: logging.warning("No similarity values calculated."); return

        logging.info(f"Calculated {similarity_values.size} pairwise similarities.")
        logging.info(f"Mean Similarity: {np.mean(similarity_values):.3f}")
        logging.info(f"Median Similarity: {np.median(similarity_values):.3f}")
        logging.info(f"Std Dev Similarity: {np.std(similarity_values):.3f}")
        logging.info(f"Min Similarity: {np.min(similarity_values):.3f}")
        logging.info(f"Max Similarity: {np.max(similarity_values):.3f}")
        logging.info(f"90th Percentile Similarity: {np.percentile(similarity_values, 90):.3f}")
        logging.info(f"95th Percentile Similarity: {np.percentile(similarity_values, 95):.3f}")
        logging.info(f"Similarity > 0.85 count: {np.sum(similarity_values > 0.85)}")
        logging.info(f"Similarity > 0.90 count: {np.sum(similarity_values > 0.90)}")


        # Plot Histogram
        plt.figure(figsize=(10, 6))
        plt.hist(similarity_values, bins=50, range=(min(0, np.min(similarity_values)), 1), edgecolor='black') # Adjust range slightly if needed
        plt.title(f'Distribution of Pairwise Description Similarities (Sample Size: {len(embeddings_sample)})')
        plt.xlabel('Cosine Similarity Score')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plot_path = os.path.join(OUTPUT_DIR, "similarity_distribution.png")
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Similarity distribution histogram saved to: {plot_path}")

    except Exception as e:
        logging.error(f"Failed during similarity analysis: {e}", exc_info=True)


def analyze_trends_at_thresholds(all_data):
    """Runs trend analysis at different rating thresholds."""
    logging.info("\n--- Trend Analysis Sensitivity ---")
    if not analysis_utils.SKLEARN_AVAILABLE:
         logging.warning("Skipping trend analysis sensitivity: scikit-learn not installed.")
         return

    # Use the analysis utils functions directly
    original_min_ideas = config.TREND_ANALYSIS_MIN_IDEAS # Store original config
    original_rating_thresh = config.RATING_THRESHOLD

    for threshold in TREND_THRESHOLDS_TO_TEST:
        logging.info(f"\nAnalyzing trends for ideas with rating >= {threshold}...")
        # Filter data for this threshold
        filtered_data = [item for item in all_data if item.get('rating') is not None and item['rating'] >= threshold]

        # Temporarily override config for the analysis function call
        config.TREND_ANALYSIS_MIN_IDEAS = 1 # Set min to 1 for testing thresholds
        config.RATING_THRESHOLD = threshold # Set threshold for this run

        if not filtered_data:
             logging.warning(f" Threshold {threshold}: No ideas found meeting this threshold.")
             continue

        logging.info(f" Threshold {threshold}: Running combined analysis on {len(filtered_data)} ideas.")
        # Pass the already filtered data
        themes = analysis_utils.get_combined_themes(filtered_data)
        logging.info(f" Threshold {threshold}: Combined Themes Found: {themes}")

    # Restore original config values
    config.TREND_ANALYSIS_MIN_IDEAS = original_min_ideas
    config.RATING_THRESHOLD = original_rating_thresh


# --- Main ---
if __name__ == "__main__":
    logging.info("Starting Tuning Analyzer...")
    # Ensure DB exists and schema is potentially updated
    try:
        state_manager.init_db()
    except Exception as e:
        logging.critical(f"Failed to initialize database. Exiting. Error: {e}")
        sys.exit(1)

    # Load data
    idea_data = load_all_idea_data()

    if not idea_data:
        logging.warning("No data loaded from database. Cannot perform analysis.")
        sys.exit(1)

    # Run analyses
    analyze_ratings(idea_data)
    analyze_similarity(idea_data)
    analyze_trends_at_thresholds(idea_data)

    logging.info("\nTuning analysis complete.")
    logging.info(f"Check the '{OUTPUT_DIR}' directory for plots (rating_distribution.png, similarity_distribution.png).")
    logging.info("Use these results to inform adjustments to RATING_THRESHOLD, NEGATIVE_FEEDBACK_SIMILARITY_THRESHOLD, and TREND_ANALYSIS_* settings in config.py or .env")