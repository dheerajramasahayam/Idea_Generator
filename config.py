import os
import logging
import json
import random
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Keys ---
# Search API - Load all potential keys, validation will check which one is active
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY")

# Determine active search provider (simple priority: Brave > Serper > Google)
SEARCH_PROVIDER = None
if BRAVE_API_KEY and BRAVE_API_KEY != "YOUR_BRAVE_API_KEY":
    SEARCH_PROVIDER = "brave"; logging.info("Using Brave Search API.")
elif SERPER_API_KEY and SERPER_API_KEY != "YOUR_SERPER_API_KEY":
    SEARCH_PROVIDER = "serper"; logging.info("Using Serper API.")
elif GOOGLE_API_KEY and GOOGLE_API_KEY != "YOUR_GOOGLE_API_KEY" and GOOGLE_CSE_ID and GOOGLE_CSE_ID != "YOUR_GOOGLE_CSE_ID":
    SEARCH_PROVIDER = "google"; logging.info("Using Google Custom Search API.")
else: logging.warning("No Search API Key found or configured correctly in .env")

# Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-pro")
UPTIME_KUMA_PUSH_URL = os.environ.get("UPTIME_KUMA_PUSH_URL")

# GitHub API (Optional, for richer search on dev tools)
GITHUB_PAT = os.environ.get("GITHUB_PAT") # Personal Access Token

# --- Email Notification Settings ---
ENABLE_EMAIL_NOTIFICATIONS = os.environ.get("ENABLE_EMAIL_NOTIFICATIONS", "false").lower() == "true"
SMTP_SERVER = os.environ.get("SMTP_SERVER")
SMTP_PORT = os.environ.get("SMTP_PORT", 587)
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
EMAIL_SENDER = os.environ.get("EMAIL_SENDER")
EMAIL_RECIPIENT = os.environ.get("EMAIL_RECIPIENT")

# --- Embedding Settings ---
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
NEGATIVE_FEEDBACK_SIMILARITY_THRESHOLD = float(os.environ.get("NEGATIVE_FEEDBACK_SIMILARITY_THRESHOLD", 0.75)) # Tuned

# --- Trend Analysis Settings ---
TREND_ANALYSIS_MIN_IDEAS = int(os.environ.get("TREND_ANALYSIS_MIN_IDEAS", 10))
TREND_ANALYSIS_RUN_INTERVAL = int(os.environ.get("TREND_ANALYSIS_RUN_INTERVAL", 5))
TREND_NGRAM_COUNT = int(os.environ.get("TREND_NGRAM_COUNT", 3))
TREND_LDA_TOPICS = int(os.environ.get("TREND_LDA_TOPICS", 2)) # Tuned
TREND_LDA_WORDS = int(os.environ.get("TREND_LDA_WORDS", 3))
TREND_CLUSTER_COUNT = int(os.environ.get("TREND_CLUSTER_COUNT", 2)) # Tuned
TREND_CLUSTER_THEMES_PER_CLUSTER = int(os.environ.get("TREND_CLUSTER_THEMES_PER_CLUSTER", 1))

# --- Idea Variation Settings ---
ENABLE_VARIATION_GENERATION = os.environ.get("ENABLE_VARIATION_GENERATION", "true").lower() == "true"
VARIATION_GENERATION_PROBABILITY = float(os.environ.get("VARIATION_GENERATION_PROBABILITY", 0.25))
VARIATION_SOURCE_MIN_RATING = float(os.environ.get("VARIATION_SOURCE_MIN_RATING", 7.0))
VARIATION_SOURCE_MAX_RATING = float(os.environ.get("VARIATION_SOURCE_MAX_RATING", 7.4)) # Adjusted based on threshold

# --- Multi-Step Generation Settings ---
ENABLE_MULTI_STEP_GENERATION = os.environ.get("ENABLE_MULTI_STEP_GENERATION", "false").lower() == "true"
NUM_CONCEPTS_TO_GENERATE = int(os.environ.get("NUM_CONCEPTS_TO_GENERATE", 5))
NUM_CONCEPTS_TO_SELECT = int(os.environ.get("NUM_CONCEPTS_TO_SELECT", 2))
NUM_IDEAS_PER_CONCEPT = int(os.environ.get("NUM_IDEAS_PER_CONCEPT", 5))

# --- Focused Re-generation Settings ---
ENABLE_FOCUSED_REGENERATION = os.environ.get("ENABLE_FOCUSED_REGENERATION", "false").lower() == "true" # Disabled by default
REGENERATION_TRIGGER_THRESHOLD = float(os.environ.get("REGENERATION_TRIGGER_THRESHOLD", 5.0)) # Trigger if score is below this
NUM_REGENERATION_ATTEMPTS = int(os.environ.get("NUM_REGENERATION_ATTEMPTS", 1)) # Generate 1 alternative

# --- Dynamic Prompt Selection Settings ---
ENABLE_DYNAMIC_PROMPT_SELECTION = os.environ.get("ENABLE_DYNAMIC_PROMPT_SELECTION", "false").lower() == "true" # Disabled by default
DYNAMIC_SELECTION_MIN_DATA = int(os.environ.get("DYNAMIC_SELECTION_MIN_DATA", 10)) # Min ideas per type before using its avg rating
DYNAMIC_SELECTION_TEMP = float(os.environ.get("DYNAMIC_SELECTION_TEMP", 0.1)) # Softmax temperature (lower = more greedy)

# --- Automated Example Generation Settings ---
ENABLE_AUTO_EXAMPLE_GENERATION = os.environ.get("ENABLE_AUTO_EXAMPLE_GENERATION", "false").lower() == "true" # Disabled by default
AUTO_EXAMPLE_RUN_INTERVAL = int(os.environ.get("AUTO_EXAMPLE_RUN_INTERVAL", 20)) # How often (in runs) to generate new examples
AUTO_EXAMPLE_SOURCE_COUNT = int(os.environ.get("AUTO_EXAMPLE_SOURCE_COUNT", 10)) # How many top ideas to use as source
AUTO_EXAMPLE_TARGET_COUNT = int(os.environ.get("AUTO_EXAMPLE_TARGET_COUNT", 3)) # How many new examples to generate

# --- Script Parameters ---
try:
    IDEAS_PER_BATCH = int(os.environ.get("IDEAS_PER_BATCH", 10))
    RATING_THRESHOLD = float(os.environ.get("RATING_THRESHOLD", 7.5)) # Keep lowered threshold
    SEARCH_RESULTS_LIMIT = int(os.environ.get("SEARCH_RESULTS_LIMIT", 10))
    DELAY_BETWEEN_IDEAS = int(os.environ.get("DELAY_BETWEEN_IDEAS", 5))
    MAX_CONCURRENT_TASKS = int(os.environ.get("MAX_CONCURRENT_TASKS", 1))
    MAX_SUMMARY_LENGTH = int(os.environ.get("MAX_SUMMARY_LENGTH", 2500))
    MAX_RUNS = int(os.environ.get("MAX_RUNS", 999999)) # Default for continuous running
    WAIT_BETWEEN_BATCHES = int(os.environ.get("WAIT_BETWEEN_BATCHES", 10))
    EXPLORE_RATIO = float(os.environ.get("EXPLORE_RATIO", 0.2))
    # Ensure SMTP_PORT is treated as integer
    smtp_port_str = os.environ.get("SMTP_PORT", "587")
    SMTP_PORT = int(smtp_port_str) if smtp_port_str.isdigit() else 587
except ValueError as e:
    logging.error(f"Error parsing numeric config: {e}. Using defaults.")
    IDEAS_PER_BATCH = 10; RATING_THRESHOLD = 7.5; SEARCH_RESULTS_LIMIT = 10
    DELAY_BETWEEN_IDEAS = 5; MAX_CONCURRENT_TASKS = 1; MAX_SUMMARY_LENGTH = 2500
    MAX_RUNS = 999999; WAIT_BETWEEN_BATCHES = 10; EXPLORE_RATIO = 0.2
    SMTP_PORT = 587; NEGATIVE_FEEDBACK_SIMILARITY_THRESHOLD = 0.75
    TREND_ANALYSIS_MIN_IDEAS = 10; TREND_ANALYSIS_RUN_INTERVAL = 5
    TREND_NGRAM_COUNT = 3; TREND_LDA_TOPICS = 2; TREND_LDA_WORDS = 3
    TREND_CLUSTER_COUNT = 2; TREND_CLUSTER_THEMES_PER_CLUSTER = 1
    VARIATION_GENERATION_PROBABILITY = 0.25; VARIATION_SOURCE_MIN_RATING = 7.0
    VARIATION_SOURCE_MAX_RATING = 7.4; NUM_VARIATIONS_TO_GENERATE = 5
    NUM_CONCEPTS_TO_GENERATE = 5; NUM_CONCEPTS_TO_SELECT = 2; NUM_IDEAS_PER_CONCEPT = 5
    REGENERATION_TRIGGER_THRESHOLD = 5.0; NUM_REGENERATION_ATTEMPTS = 1
    DYNAMIC_SELECTION_MIN_DATA = 10; DYNAMIC_SELECTION_TEMP = 0.1
    AUTO_EXAMPLE_RUN_INTERVAL = 20; AUTO_EXAMPLE_SOURCE_COUNT = 10; AUTO_EXAMPLE_TARGET_COUNT = 3


# --- Rating Weights ---
RATING_WEIGHTS = { "need": 0.25, "willingnesstopay": 0.30, "competition": 0.10, "monetization": 0.20, "feasibility": 0.15 }
total_weight = sum(RATING_WEIGHTS.values())
if total_weight > 0: RATING_WEIGHTS = {k: v / total_weight for k, v in RATING_WEIGHTS.items()}
else: num_criteria = len(RATING_WEIGHTS); RATING_WEIGHTS = {k: 1.0 / num_criteria for k in RATING_WEIGHTS.keys()}

# --- File Paths ---
OUTPUT_FILE = "gemini_rated_ideas.md"; STATE_FILE = "ideas_state.db"; LOG_FILE = "automator.log"
PROMPT_FILE = "prompts.json"
GENERATED_EXAMPLES_FILE = "generated_examples.json" # File to store dynamic examples

# --- Load Prompts from JSON ---
_prompts = {} # Initialize empty
try:
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        _prompts = json.load(f)
    logging.info(f"Successfully loaded prompts from {PROMPT_FILE}")
except FileNotFoundError:
    logging.error(f"Prompt file '{PROMPT_FILE}' not found. Using empty prompts.")
except json.JSONDecodeError as e:
    logging.error(f"Error decoding JSON from '{PROMPT_FILE}': {e}. Using empty prompts.")
except Exception as e:
     logging.error(f"Unexpected error loading prompts from '{PROMPT_FILE}': {e}. Using empty prompts.")

# Assign prompts to variables & create mapping for dynamic selection
IDEA_GENERATION_PROMPT_TEMPLATES = []
PROMPT_TYPE_MAP = {} # Map template string back to type identifier

def _add_prompt(key, identifier):
    prompt = _prompts.get(key)
    if prompt:
        IDEA_GENERATION_PROMPT_TEMPLATES.append(prompt)
        PROMPT_TYPE_MAP[prompt] = identifier
    else:
        logging.warning(f"Prompt key '{key}' not found in {PROMPT_FILE}.")

_add_prompt("IDEA_GENERATION_GENERAL", "general")
_add_prompt("IDEA_GENERATION_PLATFORM", "platform")
_add_prompt("IDEA_GENERATION_DATA", "data")
_add_prompt("IDEA_GENERATION_AI", "ai")

if not IDEA_GENERATION_PROMPT_TEMPLATES:
     logging.error("No valid IDEA_GENERATION prompts loaded! Using fallback.")
     fallback_prompt = "Generate {num_ideas} SaaS ideas."
     IDEA_GENERATION_PROMPT_TEMPLATES = [fallback_prompt]
     PROMPT_TYPE_MAP[fallback_prompt] = "fallback"

IDEA_DESCRIPTION_PROMPT_TEMPLATE = _prompts.get("IDEA_DESCRIPTION", "")
IDEA_VARIATION_PROMPT_TEMPLATE = _prompts.get("IDEA_VARIATION", "")
SEARCH_QUERY_GENERATION_PROMPT_TEMPLATE = _prompts.get("SEARCH_QUERY_GENERATION", "")
FACT_EXTRACTION_PROMPT_TEMPLATE = _prompts.get("FACT_EXTRACTION", "")
RATING_PROMPT_TEMPLATE = _prompts.get("RATING", "")
SELF_CRITIQUE_PROMPT_TEMPLATE = _prompts.get("SELF_CRITIQUE", "")
CONCEPT_GENERATION_PROMPT_TEMPLATE = _prompts.get("CONCEPT_GENERATION", "")
CONCEPT_SELECTION_PROMPT_TEMPLATE = _prompts.get("CONCEPT_SELECTION", "")
SPECIFIC_IDEA_GENERATION_PROMPT_TEMPLATE = _prompts.get("SPECIFIC_IDEA_GENERATION", "")
IDEA_REGENERATION_PROMPT_TEMPLATE = _prompts.get("IDEA_REGENERATION", "")
EXAMPLE_SYNTHESIS_PROMPT_TEMPLATE = _prompts.get("EXAMPLE_SYNTHESIS", "") # Load example synthesis prompt


# --- Validation ---
def validate_config():
    """Checks if essential API keys, prompts, and required email settings are loaded."""
    valid = True
    if not SEARCH_PROVIDER: logging.error("No valid Search API provider configured."); valid = False
    elif SEARCH_PROVIDER == "google" and (not GOOGLE_API_KEY or not GOOGLE_CSE_ID): logging.error("Google Search keys missing."); valid = False
    elif SEARCH_PROVIDER == "serper" and not SERPER_API_KEY: logging.error("Serper Search key missing."); valid = False
    elif SEARCH_PROVIDER == "brave" and (not BRAVE_API_KEY or BRAVE_API_KEY == "YOUR_BRAVE_API_KEY"): logging.error("Brave Search key missing or placeholder."); valid = False # Added placeholder check

    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY": logging.error("GEMINI_API_KEY missing or placeholder."); valid = False # Added placeholder check
    if not IDEA_GENERATION_PROMPT_TEMPLATES: logging.error("No IDEA_GENERATION prompts loaded."); valid = False
    if not all([IDEA_DESCRIPTION_PROMPT_TEMPLATE, SEARCH_QUERY_GENERATION_PROMPT_TEMPLATE, FACT_EXTRACTION_PROMPT_TEMPLATE, RATING_PROMPT_TEMPLATE, SELF_CRITIQUE_PROMPT_TEMPLATE]):
         logging.error("One or more essential prompt templates failed to load."); valid = False
    if ENABLE_VARIATION_GENERATION and not IDEA_VARIATION_PROMPT_TEMPLATE: logging.error("Variation generation enabled, but prompt missing."); valid = False
    if ENABLE_MULTI_STEP_GENERATION and not all([CONCEPT_GENERATION_PROMPT_TEMPLATE, CONCEPT_SELECTION_PROMPT_TEMPLATE, SPECIFIC_IDEA_GENERATION_PROMPT_TEMPLATE]):
         logging.error("Multi-step generation enabled, but prompts missing."); valid = False
    if ENABLE_FOCUSED_REGENERATION and not IDEA_REGENERATION_PROMPT_TEMPLATE:
         logging.error("Focused re-generation enabled, but IDEA_REGENERATION prompt missing.")
         valid = False
    if ENABLE_AUTO_EXAMPLE_GENERATION and not EXAMPLE_SYNTHESIS_PROMPT_TEMPLATE: # Check example synthesis prompt
         logging.error("Auto example generation enabled, but EXAMPLE_SYNTHESIS prompt missing.")
         valid = False
    if ENABLE_EMAIL_NOTIFICATIONS:
        logging.info("Email notifications enabled. Validating SMTP settings...")
        email_valid = True
        if not SMTP_SERVER: logging.error("SMTP_SERVER not set."); email_valid = False
        if not SMTP_USER: logging.error("SMTP_USER not set."); email_valid = False
        if not SMTP_PASSWORD: logging.error("SMTP_PASSWORD not set."); email_valid = False
        if not EMAIL_SENDER: logging.error("EMAIL_SENDER not set."); email_valid = False
        if not EMAIL_RECIPIENT: logging.error("EMAIL_RECIPIENT not set."); email_valid = False
        try: int(SMTP_PORT)
        except (ValueError, TypeError): logging.error(f"Invalid SMTP_PORT: {SMTP_PORT}."); email_valid = False
        if not email_valid: valid = False
    if EMBEDDING_MODEL == "all-MiniLM-L6-v2": logging.info(f"Using default embedding model: {EMBEDDING_MODEL}")
    else: logging.info(f"Using embedding model from env: {EMBEDDING_MODEL}")
    if not GITHUB_PAT: logging.warning("GITHUB_PAT not set in .env. GitHub API calls will be rate-limited.")
    return valid

# --- Logging Setup ---
def setup_logging():
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[ logging.FileHandler(LOG_FILE, encoding='utf-8'), logging.StreamHandler() ])

# Initial setup call
setup_logging()
