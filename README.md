# SaaS Idea Automator

This project contains a Python script designed to automate the process of generating, researching, and evaluating potential SaaS (Software as a Service) ideas, with a focus on identifying concepts with strong day-1 revenue potential. It leverages Google's Gemini API for idea generation, query generation, and rating, and a configurable search API (currently set to Brave Search API) for research.

## Core Functionality

1.  **Idea Generation:** Uses the Gemini API to generate batches of SaaS ideas based on configurable prompts, including negative feedback (avoiding previously processed or low-rated ideas) and optional positive theme focusing (explore/exploit).
2.  **Self-Critique:** Performs an AI-driven critique step on generated ideas to filter for alignment with core goals (B2B/prosumer focus, day-1 potential, specific types) before proceeding with research.
3.  **Targeted Research:** For each promising idea, it uses the Gemini API to generate specific search queries focused on competition, pricing, and market need. It then uses the Google Custom Search API to gather relevant web search results.
4.  **AI-Powered Rating:** Analyzes the research summary using the Gemini API based on defined criteria (Need, WillingnessToPay, Competition, Monetization, Feasibility) defined in `config.py`. It extracts both a score and a justification for each criterion.
5.  **Weighted Scoring:** Calculates a final weighted score based on configurable weights (`RATING_WEIGHTS` in `config.py`) assigned to each rating criterion.
6.  **Output:** Saves ideas that meet a configurable score threshold (`RATING_THRESHOLD` in `config.py`) to an output markdown file (`gemini_rated_ideas.md`), including the weighted score and the AI's justifications for each criterion.
7.  **State Management:** Uses an SQLite database (`ideas_state.db`) to reliably track processed ideas and their status/ratings, preventing reprocessing and enabling features like negative feedback.
8.  **Monitoring Hook:** Includes integration for push-based monitoring services like Uptime Kuma via a configurable URL (`UPTIME_KUMA_PUSH_URL` in `.env`).
9.  **Trend Analysis:** Includes a separate script (`trend_analyzer.py`) to perform basic keyword frequency analysis on the high-scoring ideas saved in the output file.

## Project Structure

```
.
├── .env                    # Environment variables (API Keys, Config - MUST BE CREATED)
├── .gitignore              # Git ignore rules
├── api_clients.py          # Functions for interacting with external APIs (Gemini, Search Provider, Uptime Kuma)
├── config.py               # Loads .env, defines constants, prompts, weights
├── ideas_state.db          # SQLite database for tracking processed ideas (created on first run)
├── saas_idea_automator.py  # Main script orchestrating the process
├── state_manager.py        # Functions for interacting with the SQLite state database
├── requirements.txt        # Python dependencies
├── trend_analyzer.py       # Script to analyze trends in high-scoring ideas
├── automator.log           # Main log file
├── gemini_rated_ideas.md   # Output file for high-scoring ideas (created on first run)
└── README.md               # This file
```

## Setup

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/dheerajramasahayam/Idea_Generator.git
    cd Idea_Generator
    ```
2.  **Create Virtual Environment:** (Recommended)
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(This installs `aiohttp`, `python-dotenv`, `requests`)*
4.  **Create and Configure `.env` File:**
    *   Copy or rename `.env.example` (if provided) or create a new file named `.env` in the project root.
    *   Add your API keys. Choose ONE search provider (Brave, Serper, or Google) and provide its key(s). Brave is currently prioritized in `config.py`.
        ```dotenv
        # --- Search API (Choose ONE) ---
        BRAVE_API_KEY="YOUR_BRAVE_API_KEY"
        # SERPER_API_KEY="YOUR_SERPER_API_KEY"
        # GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
        # GOOGLE_CSE_ID="YOUR_GOOGLE_CSE_ID"

        # --- AI Model API ---
        GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
        # Optional: Specify model
        # GEMINI_MODEL="gemini-pro"

        # --- Monitoring ---
        # Optional: Add your Uptime Kuma Push URL if using it
        # UPTIME_KUMA_PUSH_URL="YOUR_PUSH_URL_HERE"

        # --- Email Notifications ---
        # Optional: Enable and configure SMTP settings
        # ENABLE_EMAIL_NOTIFICATIONS="true"
        # SMTP_SERVER="smtp.example.com"
        # SMTP_PORT="587"
        # SMTP_USER="your_email@example.com"
        # SMTP_PASSWORD="YOUR_EMAIL_APP_PASSWORD"
        # EMAIL_SENDER="Sender Name <your_email@example.com>"
        # EMAIL_RECIPIENT="recipient@example.com"

        # Optional: Override parameters from config.py
        # GEMINI_MODEL="gemini-1.5-pro-latest"
        # MAX_RUNS=50
        # RATING_THRESHOLD=8.0
        ```
    *   **Important:** Ensure the `.env` file is included in your `.gitignore` and never committed to version control.
5.  **Review Configuration (`config.py`):** Check default parameters like `MAX_RUNS`, `IDEAS_PER_BATCH`, `RATING_THRESHOLD`, `RATING_WEIGHTS`, etc., and override them in `.env` if needed.

## Running the Automator

Ensure your virtual environment is activated (`source venv/bin/activate`).

**Direct Execution (for testing):**

```bash
python3 saas_idea_automator.py
```

The script will run for the number of batches specified by `MAX_RUNS` in `config.py`. Logs will be printed to the console and saved to `automator.log`. High-scoring ideas will be appended to `gemini_rated_ideas.md`. Processed ideas are tracked in `ideas_state.db`.

**Continuous Running (Server Deployment using `systemd`):**

Refer to the deployment instructions provided previously or the example `saas-automator.service` file (you'll need to create this file on your server at `/etc/systemd/system/`).

Key `systemd` commands:
*   `sudo systemctl daemon-reload` (After creating/editing the service file)
*   `sudo systemctl enable saas-automator.service` (Start on boot)
*   `sudo systemctl start saas-automator.service` (Start now)
*   `sudo systemctl status saas-automator.service` (Check status)
*   `sudo systemctl stop saas-automator.service` (Stop the service)
*   `sudo journalctl -u saas-automator.service -f` (Follow systemd logs)

## Running the Trend Analyzer

After the main script has run for some time and populated `gemini_rated_ideas.md` with high-scoring ideas:

```bash
python3 trend_analyzer.py
```

This will read the output file and print the most common keywords found in the names of the successful ideas.

## Dependencies

*   Python 3.x
*   Libraries listed in `requirements.txt`:
    *   `aiohttp`: For asynchronous HTTP requests.
    *   `python-dotenv`: For loading the `.env` file.
*   `requests`: Standard HTTP library.
*   `nltk`: For text processing (tokenization, stopwords).
*   `sentence-transformers`: For generating semantic embeddings (requires PyTorch or TensorFlow).
*   `scikit-learn`: For LDA topic modeling and K-Means clustering.
*   `numpy`: Dependency for scikit-learn and sentence-transformers.
*   `sqlite3`: (Included in standard Python library)

*(Note: The script attempts to download required NLTK data (`punkt`, `stopwords`) automatically on first run if missing).*
