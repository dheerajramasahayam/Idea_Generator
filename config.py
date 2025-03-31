import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Keys ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-pro")
UPTIME_KUMA_PUSH_URL = os.environ.get("UPTIME_KUMA_PUSH_URL") # Load Push URL

# --- Script Parameters ---
# Use environment variables or fall back to defaults
try:
    IDEAS_PER_BATCH = int(os.environ.get("IDEAS_PER_BATCH", 10))
    RATING_THRESHOLD = float(os.environ.get("RATING_THRESHOLD", 9.0)) # Increased threshold to 9.0
    SEARCH_RESULTS_LIMIT = int(os.environ.get("SEARCH_RESULTS_LIMIT", 10)) # Increased
    DELAY_BETWEEN_IDEAS = int(os.environ.get("DELAY_BETWEEN_IDEAS", 5))
    MAX_CONCURRENT_TASKS = int(os.environ.get("MAX_CONCURRENT_TASKS", 1))
    MAX_SUMMARY_LENGTH = int(os.environ.get("MAX_SUMMARY_LENGTH", 2500)) # Increased
    MAX_RUNS = int(os.environ.get("MAX_RUNS", 999999)) # Set high default for continuous running
    WAIT_BETWEEN_BATCHES = int(os.environ.get("WAIT_BETWEEN_BATCHES", 10))
    EXPLORE_RATIO = float(os.environ.get("EXPLORE_RATIO", 0.2)) # 20% chance to explore (broad prompt)
except ValueError as e:
    logging.error(f"Error parsing numeric config from environment variables: {e}. Using defaults.")
    IDEAS_PER_BATCH = 10
    RATING_THRESHOLD = 9.0 # Increased threshold default to 9.0
    SEARCH_RESULTS_LIMIT = 10 # Increased
    DELAY_BETWEEN_IDEAS = 5
    MAX_CONCURRENT_TASKS = 1
    MAX_SUMMARY_LENGTH = 2500 # Increased
    MAX_RUNS = 999999 # Set high default for continuous running
    WAIT_BETWEEN_BATCHES = 10
    EXPLORE_RATIO = 0.2 # Default explore ratio

# --- Rating Weights ---
# Define weights for each criterion. Adjusted to prioritize day-1 revenue potential.
RATING_WEIGHTS = {
    "need": 0.25,             # Still important
    "willingnesstopay": 0.30, # Increased weight
    "competition": 0.10,      # Decreased weight
    "monetization": 0.20,     # Increased weight
    "feasibility": 0.15       # Decreased weight
}
# Normalize weights to ensure they sum to 1 (though the suggested values already do)
total_weight = sum(RATING_WEIGHTS.values())
if total_weight > 0:
    RATING_WEIGHTS = {k: v / total_weight for k, v in RATING_WEIGHTS.items()}
else: # Fallback to equal weights if sum is zero
     num_criteria = len(RATING_WEIGHTS)
     RATING_WEIGHTS = {k: 1.0 / num_criteria for k in RATING_WEIGHTS.keys()}


# --- File Paths ---
OUTPUT_FILE = "gemini_rated_ideas.md"
STATE_FILE = "ideas_state.db" # Changed to SQLite database file
LOG_FILE = "automator.log"

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

# Updated rating prompt asking for individual criteria scores AND justifications
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
    """Checks if essential API keys are loaded."""
    valid = True
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY":
        logging.error("GOOGLE_API_KEY not found or not set in .env file.")
        valid = False
    if not GOOGLE_CSE_ID or GOOGLE_CSE_ID == "YOUR_GOOGLE_CSE_ID":
        logging.error("GOOGLE_CSE_ID not found or not set in .env file.")
        valid = False
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        logging.error("GEMINI_API_KEY not found or not set in .env file.")
        valid = False
    # Optional: Validate Kuma URL format roughly, but mainly check if it exists if needed
    # if not UPTIME_KUMA_PUSH_URL:
    #     logging.warning("UPTIME_KUMA_PUSH_URL not set in .env file. Monitoring ping will be disabled.")
        # valid = False # Decide if Kuma URL is mandatory
    return valid

# --- Logging Setup ---
def setup_logging():
    # Remove existing handlers to avoid duplicate logs if script is re-run
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

# Initial setup call
setup_logging()
