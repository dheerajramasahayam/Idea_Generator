# SaaS Idea Automator

This project contains a Python script designed to automate the process of generating, researching, and evaluating potential SaaS (Software as a Service) ideas, with a focus on identifying concepts with strong day-1 revenue potential. It leverages Google's Gemini API for AI tasks and configurable search APIs (Brave Search default, optional GitHub) for research.

## Core Functionality

1.  **Idea Generation:** Uses multiple strategies (diverse prompts, variation on existing ideas, multi-step concept refinement) via the Gemini API. Includes negative feedback (keyword/example avoidance in prompts).
2.  **Description Generation:** Creates short descriptions for each idea name for better semantic understanding.
3.  **Self-Critique:** Performs an AI-driven critique step on generated ideas to filter for alignment with core goals.
4.  **Targeted Research:**
    *   Generates specific search queries using AI, encouraging operator use.
    *   Uses a configurable web search API (Brave default, with freshness/result filters).
    *   **Optionally searches GitHub repositories** for developer-related ideas to find existing tools/libraries.
5.  **AI-Powered Evaluation:**
    *   Extracts key facts related to rating criteria from the combined research summary.
    *   Rates the idea based strictly on extracted facts using a conservative rubric.
6.  **Weighted Scoring:** Calculates a final weighted score based on configurable weights.
7.  **Output & State:** Saves ideas meeting a configurable score threshold to markdown. Uses an SQLite database to track processed ideas, status, ratings, justifications, descriptions, and embeddings.
8.  **Feedback Loops:**
    *   **Semantic Negative Feedback:** Filters new ideas based on description similarity to previously low-rated ideas.
    *   **Trend Analysis:** Periodically analyzes high-rated ideas (N-grams, LDA, Clustering on descriptions) to identify promising themes.
    *   **Adaptive Explore/Exploit:** Adjusts generation strategy based on recent success rate.
9.  **Monitoring & Notifications:** Includes hooks for Uptime Kuma and conditional email summaries.
10. **Analysis Tools:** Provides a separate `tune_analyzer.py` script to analyze database results and inform parameter tuning.

## Project Structure

```
.
├── .env                    # Environment variables (API Keys, Config - MUST BE CREATED)
├── .env.example            # Example environment file
├── .gitignore              # Git ignore rules
├── api_clients.py          # Functions for interacting with external APIs
├── analysis_utils.py       # Text processing, embedding, trend analysis functions
├── config.py               # Loads .env, defines constants, weights, loads prompts
├── ideas_state.db          # SQLite database for tracking state (created on first run)
├── notifications.py        # Email notification logic
├── prompts.json            # Contains all prompt templates used by the AI
├── saas_idea_automator.py  # Main script orchestrating the process
├── state_manager.py        # Functions for interacting with the SQLite database
├── requirements.txt        # Python dependencies
├── tune_analyzer.py        # Script to analyze results and help tuning
├── analysis_output/        # Directory for plots from tune_analyzer.py (created by script)
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
4.  **Create and Configure `.env` File:**
    *   Copy `.env.example` to `.env`.
    *   Add your API keys:
        *   Choose ONE search provider (Brave recommended).
        *   Add your `GEMINI_API_KEY`.
        *   **Optional (Recommended):** Add a `GITHUB_PAT` (Personal Access Token) for GitHub searches. Create one at `https://github.com/settings/tokens` (use the "Classic" token type) with the `public_repo` scope selected. This significantly increases the rate limit for GitHub searches.
        *   Configure optional monitoring/email settings if desired.
    *   Review other settings in `.env.example` (like `ENABLE_MULTI_STEP_GENERATION`, `ENABLE_VARIATION_GENERATION`, thresholds, etc.) and override defaults in `.env` if needed.
    *   **Important:** Ensure `.env` is in your `.gitignore`.
5.  **Review Configuration (`config.py` & `prompts.json`):** Check default parameters and prompt wording. You can tune prompts directly in `prompts.json`.

## Running the Automator

Ensure your virtual environment is activated (`source venv/bin/activate`).

**Direct Execution:**

```bash
python3 saas_idea_automator.py
```

The script will run continuously by default (`MAX_RUNS=999999`). Use Ctrl+C to stop. Logs are in `automator.log`, high-scoring ideas in `gemini_rated_ideas.md`.

**Continuous Running (Server Deployment using `systemd`):**

Refer to the example `saas-automator.service` file and adapt it for your server environment.

## Running the Tuning Analyzer

After the main script has run and populated `ideas_state.db`:

```bash
python3 tune_analyzer.py
```

Review the console output and the plots saved in `analysis_output/` to inform potential adjustments to parameters in `.env` or `config.py`.

## Dependencies

*   Python 3.x
*   Libraries listed in `requirements.txt`:
    *   `aiohttp`: Async HTTP requests.
    *   `python-dotenv`: Loads `.env` file.
    *   `requests`: Standard HTTP library.
    *   `nltk`: Text processing (stopwords, ngrams).
    *   `sentence-transformers`: Semantic embeddings.
    *   `scikit-learn`: LDA and Clustering analysis.
    *   `numpy`: Numerical operations.
    *   `yarl`: URL parsing/building (used by api_clients).
    *   `matplotlib`: Plotting for analysis script.
    *   `sqlite3`: (Included in standard Python library)

*(Note: The script attempts to download required NLTK data (`stopwords`) automatically on first run if missing).*
