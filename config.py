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
NEGATIVE_FEEDBACK_SIMILARITY_THRESHOLD = float(os.environ.get("NEGATIVE_FEEDBACK_SIMILARITY_THRESHOLD", 0.85))

# --- Script Parameters ---
try:
    IDEAS_PER_BATCH = int(os.environ.get("IDEAS_PER_BATCH", 10))
    RATING_THRESHOLD = float(os.environ.get("RATING_THRESHOLD", 9.0))
    SEARCH_RESULTS_LIMIT = int(os.environ.get("SEARCH_RESULTS_LIMIT", 10))
    DELAY_BETWEEN_IDEAS = int(os.environ.get("DELAY_BETWEEN_IDEAS", 5))
    MAX_CONCURRENT_TASKS = int(os.environ.get("MAX_CONCURRENT_TASKS", 1))
    MAX_SUMMARY_LENGTH = int(os.environ.get("MAX_SUMMARY_LENGTH", 2500))
    MAX_RUNS = int(os.environ.get("MAX_RUNS", 999999))
    WAIT_BETWEEN_BATCHES = int(os.environ.get("WAIT_BETWEEN_BATCHES", 10))
    EXPLORE_RATIO = float(os.environ.get("EXPLORE_RATIO", 0.2))
    SMTP_PORT = int(SMTP_PORT)
except ValueError as e:
    logging.error(f"Error parsing numeric config: {e}. Using defaults.")
    IDEAS_PER_BATCH = 10; RATING_THRESHOLD = 9.0; SEARCH_RESULTS_LIMIT = 10
    DELAY_BETWEEN_IDEAS = 5; MAX_CONCURRENT_TASKS = 1; MAX_SUMMARY_LENGTH = 2500
    MAX_RUNS = 999999; WAIT_BETWEEN_BATCHES = 10; EXPLORE_RATIO = 0.2
    SMTP_PORT = 587; NEGATIVE_FEEDBACK_SIMILARITY_THRESHOLD = 0.85

# --- Rating Weights ---
RATING_WEIGHTS = { "need": 0.25, "willingnesstopay": 0.30, "competition": 0.10, "monetization": 0.20, "feasibility": 0.15 }
total_weight = sum(RATING_WEIGHTS.values())
if total_weight > 0: RATING_WEIGHTS = {k: v / total_weight for k, v in RATING_WEIGHTS.items()}
else: num_criteria = len(RATING_WEIGHTS); RATING_WEIGHTS = {k: 1.0 / num_criteria for k in RATING_WEIGHTS.keys()}

# --- File Paths ---
OUTPUT_FILE = "gemini_rated_ideas.md"; STATE_FILE = "ideas_state.db"; LOG_FILE = "automator.log"

# --- Prompts ---
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

FACT_EXTRACTION_PROMPT_TEMPLATE = """
Analyze the following Research Summary for the SaaS idea "{idea_name}".
Extract key facts, evidence, or mentions *strictly from the summary text* that relate to each of the 5 criteria below.
If no relevant information is found for a criterion in the summary, state "None mentioned".
Be concise. Output *only* the facts for each criterion, formatted exactly like this:

Need:
- [Fact 1 related to need/pain point]
- [Fact 2 related to need/pain point]
(or "None mentioned")

WillingnessToPay:
- [Fact 1 related to payment/pricing]
- [Fact 2 related to payment/pricing]
(or "None mentioned")

Competition:
- [Fact 1 mentioning competitors/alternatives]
- [Fact 2 mentioning competitors/alternatives]
(or "None mentioned")

Monetization:
- [Fact 1 mentioning pricing models]
- [Fact 2 mentioning pricing models]
(or "None mentioned")

Feasibility:
- [Fact 1 related to existing tech/tools]
- [Fact 2 related to existing tech/tools]
(or "None mentioned")

Research Summary:
{research_summary}
"""

# Refined rating prompt - emphasizes high bar for 9+ scores
RATING_PROMPT_TEMPLATE = """
Critically evaluate the SaaS idea "{idea_name}" based *strictly* on the following Extracted Facts.
Provide a score from 0.0 to 10.0 AND a brief (one sentence max) justification for *each* of the 5 criteria.
**Be conservative with high scores (9-10). A score of 9+ requires strong, direct evidence within the facts for that specific criterion.** For example, a high 'Need' score requires facts explicitly mentioning significant user pain or demand. High 'WillingnessToPay' requires facts mentioning existing payment for similar solutions or clear pricing evidence.
Your justification MUST reference the provided facts. If facts state "None mentioned" for a criterion, assign a low score (0-2) and state that as the justification.

1.  **Need (0-10):** How strongly do the facts indicate a clear, significant user pain point or market need? (9+ requires explicit mention of strong pain/demand).
2.  **WillingnessToPay (0-10):** Do the facts provide evidence users pay for similar solutions or mention relevant pricing? (9+ requires clear evidence of payment).
3.  **Competition (0-10):** Do the facts mention existing direct competitors? (Lower score if strong/numerous competitors mentioned; 9+ requires facts indicating very weak or no direct competition).
4.  **Monetization (0-10):** Do the facts clearly hint at viable, common pricing models (like SaaS subscriptions)? (9+ requires clear mention of established models).
5.  **Feasibility (0-10):** Do the facts strongly suggest technical feasibility based on existing tools/tech? (9+ requires clear evidence similar tech exists).

Extracted Facts:
{rating_context}

Output *only* the 5 scores and their justifications in the following exact format, each on a new line:
Need: [score] | Justification: [one sentence justification based on facts]
WillingnessToPay: [score] | Justification: [one sentence justification based on facts]
Competition: [score] | Justification: [one sentence justification based on facts]
Monetization: [score] | Justification: [one sentence justification based on facts]
Feasibility: [score] | Justification: [one sentence justification based on facts]
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
    if not SEARCH_PROVIDER: logging.error("No valid Search API provider configured."); valid = False
    elif SEARCH_PROVIDER == "google" and (not GOOGLE_API_KEY or not GOOGLE_CSE_ID): logging.error("Google Search keys missing."); valid = False
    elif SEARCH_PROVIDER == "serper" and not SERPER_API_KEY: logging.error("Serper Search key missing."); valid = False
    elif SEARCH_PROVIDER == "brave" and not BRAVE_API_KEY: logging.error("Brave Search key missing."); valid = False
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY": logging.error("GEMINI_API_KEY missing."); valid = False
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
    return valid

# --- Logging Setup ---
def setup_logging():
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[ logging.FileHandler(LOG_FILE, encoding='utf-8'), logging.StreamHandler() ])

# Initial setup call
setup_logging()
