import os
import logging
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
    SEARCH_PROVIDER = "brave"
    logging.info("Using Brave Search API.")
elif SERPER_API_KEY and SERPER_API_KEY != "YOUR_SERPER_API_KEY":
    SEARCH_PROVIDER = "serper"
    logging.info("Using Serper API.")
elif GOOGLE_API_KEY and GOOGLE_API_KEY != "YOUR_GOOGLE_API_KEY" and GOOGLE_CSE_ID and GOOGLE_CSE_ID != "YOUR_GOOGLE_CSE_ID":
    SEARCH_PROVIDER = "google"
    logging.info("Using Google Custom Search API.")
else:
    logging.warning("No Search API Key found or configured correctly in .env")


# Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-pro")
UPTIME_KUMA_PUSH_URL = os.environ.get("UPTIME_KUMA_PUSH_URL")

# --- Email Notification Settings ---
ENABLE_EMAIL_NOTIFICATIONS = os.environ.get("ENABLE_EMAIL_NOTIFICATIONS", "false").lower() == "true"
SMTP_SERVER = os.environ.get("SMTP_SERVER")
SMTP_PORT = os.environ.get("SMTP_PORT", 587)
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
EMAIL_SENDER = os.environ.get("EMAIL_SENDER")
EMAIL_RECIPIENT = os.environ.get("EMAIL_RECIPIENT")

# --- Embedding Settings ---
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2") # Default Sentence Transformer model
NEGATIVE_FEEDBACK_SIMILARITY_THRESHOLD = float(os.environ.get("NEGATIVE_FEEDBACK_SIMILARITY_THRESHOLD", 0.85)) # Cosine similarity threshold

# --- Script Parameters ---
try:
    IDEAS_PER_BATCH = int(os.environ.get("IDEAS_PER_BATCH", 10))
    RATING_THRESHOLD = float(os.environ.get("RATING_THRESHOLD", 9.0))
    SEARCH_RESULTS_LIMIT = int(os.environ.get("SEARCH_RESULTS_LIMIT", 10))
    DELAY_BETWEEN_IDEAS = int(os.environ.get("DELAY_BETWEEN_IDEAS", 5))
    MAX_CONCURRENT_TASKS = int(os.environ.get("MAX_CONCURRENT_TASKS", 1))
    MAX_SUMMARY_LENGTH = int(os.environ.get("MAX_SUMMARY_LENGTH", 2500))
    MAX_RUNS = int(os.environ.get("MAX_RUNS", 1)) # Limit to 1 run for testing embeddings
    WAIT_BETWEEN_BATCHES = int(os.environ.get("WAIT_BETWEEN_BATCHES", 10))
    EXPLORE_RATIO = float(os.environ.get("EXPLORE_RATIO", 0.2))
    SMTP_PORT = int(SMTP_PORT)
except ValueError as e:
    logging.error(f"Error parsing numeric config from environment variables: {e}. Using defaults.")
    IDEAS_PER_BATCH = 10
    RATING_THRESHOLD = 9.0
    SEARCH_RESULTS_LIMIT = 10
    DELAY_BETWEEN_IDEAS = 5
    MAX_CONCURRENT_TASKS = 1
    MAX_SUMMARY_LENGTH = 2500
    MAX_RUNS = 1 # Limit to 1 run for testing embeddings
    WAIT_BETWEEN_BATCHES = 10
    EXPLORE_RATIO = 0.2
    SMTP_PORT = 587
    NEGATIVE_FEEDBACK_SIMILARITY_THRESHOLD = 0.85

# --- Rating Weights ---
RATING_WEIGHTS = {
    "need": 0.25,
    "willingnesstopay": 0.30,
    "competition": 0.10,
    "monetization": 0.20,
    "feasibility": 0.15
}
total_weight = sum(RATING_WEIGHTS.values())
if total_weight > 0:
    RATING_WEIGHTS = {k: v / total_weight for k, v in RATING_WEIGHTS.items()}
else:
     num_criteria = len(RATING_WEIGHTS)
     RATING_WEIGHTS = {k: 1.0 / num_criteria for k in RATING_WEIGHTS.keys()}


# --- File Paths ---
OUTPUT_FILE = "gemini_rated_ideas.md"
STATE_FILE = "ideas_state.db"
LOG_FILE = "automator.log"

# --- Prompts ---
# (Keep existing prompts)
IDEA_GENERATION_PROMPT_TEMPLATE = """
Generate {num_ideas} unique SaaS ideas focused on B2B or prosumer niches with potential for day-1 revenue.
Focus on enhancing existing platforms (like Shopify, Bubble, Webflow, Airtable), niche data aggregation, or freelancer/agency automation.
Avoid ideas related to crypto, NFTs, or general consumer apps (like recipe finders or basic fitness trackers).
Good examples: "Airtable Interface Usage Analytics", "Local Commercial Real Estate Zoning Change Monitor", "Shopify App Conflict Detector".
Output only a numbered list of concise idea names, each on a new line. Do not include any preamble or explanation.
Example:
1. Idea Name One
2. Idea Name Two
"""

SEARCH_QUERY_GENERATION_PROMPT_TEMPLATE = """
Generate exactly 3 targeted Google search queries to research the day-1 revenue potential of the following SaaS idea.
Focus queries on finding alternatives/competition, pricing models, and market need/pain points.
Output only the 3 search queries, each on a new line. Do not include numbers or any other text.

SaaS Idea: {idea_name}
"""

RATING_PROMPT_TEMPLATE = """
Analyze the following SaaS idea and the provided research summary.
Based *strictly* on the provided research summary text, provide a score from 0.0 to 10.0 AND a brief (one sentence max) justification for *each* of the following 5 criteria:

1.  **Need (0-10):** How clearly does the research mention or imply a specific user pain point or market need that this idea addresses? (Higher score for clearer need).
2.  **WillingnessToPay (0-10):** Does the research suggest users currently pay for similar tools, or mention pricing for alternatives? (Higher score if payment is mentioned).
3.  **Competition (0-10):** Does the research mention existing direct competitors or alternatives? (Lower score if many strong competitors are mentioned).
4.  **Monetization (0-10):** Does the research hint at common pricing models (like SaaS subscriptions) for similar tools? (Higher score for clearer models).
5.  **Feasibility (0-10):** Does the research mention that similar tools or the required technology already exist? (Higher score if feasibility seems proven by existing tools).

SaaS Idea: {idea_name}

Research Summary:
{research_summary}

Output *only* the 5 scores and their justifications in the following exact format, each on a new line:
Need: [score] | Justification: [one sentence justification]
WillingnessToPay: [score] | Justification: [one sentence justification]
Competition: [score] | Justification: [one sentence justification]
Monetization: [score] | Justification: [one sentence justification]
Feasibility: [score] | Justification: [one sentence justification]

Example:
Need: 8.5 | Justification: Research mentions users struggling with complex workflows.
WillingnessToPay: 7.0 | Justification: Pricing for alternative platform X is mentioned.
Competition: 4.0 | Justification: Only one direct competitor mentioned in the summary.
Monetization: 8.0 | Justification: Summary mentions SaaS subscription models are common.
Feasibility: 7.5 | Justification: Research shows similar tools exist, proving feasibility.

Do not include any other text, explanation, or preamble.
"""

SELF_CRITIQUE_PROMPT_TEMPLATE = """
Review the following list of generated SaaS ideas. Identify and return *only* the ideas that strongly align with these core goals:
- Focus: B2B or prosumer niches.
- Potential: Clear path to day-1 revenue (e.g., solving urgent business pain, high perceived value).
- Type: Enhancing existing platforms (Shopify, Bubble, etc.), niche data aggregation, or freelancer/agency automation.
- Avoid: Crypto, NFTs, general consumer apps (fitness, recipes).

Generated Ideas List:
{idea_list_str}

Output *only* a numbered list of the idea names from the input list that strongly align with ALL the above goals. If none strongly align, output "None". Do not include any preamble, explanation, or justification.
"""


# --- Validation ---
def validate_config():
    """Checks if essential API keys and required email settings (if enabled) are loaded."""
    valid = True
    # Check Search Provider
    if not SEARCH_PROVIDER:
         logging.error("No valid Search API provider configured in .env (checked BRAVE_API_KEY, SERPER_API_KEY, GOOGLE_API_KEY/GOOGLE_CSE_ID).")
         valid = False
    elif SEARCH_PROVIDER == "google" and (not GOOGLE_API_KEY or not GOOGLE_CSE_ID):
         logging.error("Google Search provider selected but GOOGLE_API_KEY or GOOGLE_CSE_ID missing.")
         valid = False
    elif SEARCH_PROVIDER == "serper" and not SERPER_API_KEY:
         logging.error("Serper Search provider selected but SERPER_API_KEY missing.")
         valid = False
    elif SEARCH_PROVIDER == "brave" and not BRAVE_API_KEY:
         logging.error("Brave Search provider selected but BRAVE_API_KEY missing.")
         valid = False

    # Check Gemini Key
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        logging.error("GEMINI_API_KEY not found or not set in .env file.")
        valid = False

    # Check Email Settings if Enabled
    if ENABLE_EMAIL_NOTIFICATIONS:
        logging.info("Email notifications enabled. Validating SMTP settings...")
        email_valid = True
        if not SMTP_SERVER: logging.error("SMTP_SERVER not set (required for email)."); email_valid = False
        if not SMTP_USER: logging.error("SMTP_USER not set (required for email)."); email_valid = False
        if not SMTP_PASSWORD: logging.error("SMTP_PASSWORD not set (required for email)."); email_valid = False
        if not EMAIL_SENDER: logging.error("EMAIL_SENDER not set (required for email)."); email_valid = False
        if not EMAIL_RECIPIENT: logging.error("EMAIL_RECIPIENT not set (required for email)."); email_valid = False
        try: int(SMTP_PORT)
        except (ValueError, TypeError): logging.error(f"Invalid SMTP_PORT: {SMTP_PORT}."); email_valid = False
        if not email_valid: valid = False

    # Check Embedding Model (just log)
    if EMBEDDING_MODEL == "all-MiniLM-L6-v2": logging.info(f"Using default embedding model: {EMBEDDING_MODEL}")
    else: logging.info(f"Using embedding model from env: {EMBEDDING_MODEL}")

    return valid

# --- Logging Setup ---
def setup_logging():
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[ logging.FileHandler(LOG_FILE, encoding='utf-8'), logging.StreamHandler() ]
    )

# Initial setup call
setup_logging()
