import asyncio
import aiohttp
import json
import time
import random
import logging
import re
import sys
import os
from collections import Counter # For trend analysis

# Import modular components
import config
import api_clients
import state_manager # Uses SQLite now

# --- Trend Analysis Helpers --- (Adapted from trend_analyzer.py)

# Basic list of common English stop words + some generic terms
STOP_WORDS = set([
    "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just", "don", "should", "now", "d", "ll", "m",
    "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn",
    "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn",
    "weren", "won", "wouldn", "tool", "app", "software", "platform", "manager",
    "tracker", "analyzer", "generator", "based", "using", "via", "like", "e.g.",
    "for", "with", "and", "new", "get", "use", "best", "top", "simple", "advanced",
    "idea", "ideas", "saas", "system", "service", "solution", "data", "automation",
    "niche", "focus", "specific", "regional", "monitor", "detector", "predictor",
    "comparator", "estimator", "profiler", "visualizer", "aggregator", "calculator",
    "assistant", "engine", "checker", "enhancer", "optimizer", "ai", "api", "web",
    "online", "digital", "client", "customer", "user", "business", "agency",
    "freelancer", "marketing", "sales", "content", "seo", "e-commerce", "shop",
    "store", "product", "listing", "performance", "report", "reporting", "analysis",
    "analytics", "management", "generation", "generator", "request", "collection",
    "sync", "code", "fee", "tax", "rate", "value", "cost", "revenue", "roi",
    "time", "page", "workflow", "schema", "change", "impact", "dependency",
    "uptime", "latency", "error", "pattern", "log", "file", "coverage", "debt",
    "feature", "release", "tender", "keyword", "alert", "plan", "pricing",
    "competitor", "comparison", "trend", "growth", "sentiment", "communication",
    "asset", "scope", "creep", "proposal", "domain", "backlink", "publication",
    "sequence", "email", "affiliate", "link", "onboarding", "compliance",
    "subcontractor", "real", "estate", "zoning", "commercial", "local", "vc",
    "investment", "regulatory", "fintech", "indie", "hacker", "launch", "conference",
    "sponsor", "exhibitor", "clv", "resource", "allocation", "pipeline", "liability",
    "marketplace", "testimonial", "video", "capture", "w9", "contract", "etsy",
    "shopify", "amazon", "fba", "storage", "ad", "ebay", "offer", "strategy",
    "return", "inventory", "discount", "funnel", "drop-off", "crm", "social", "media",
    "engagement", "gap", "landing", "test", "bubble", "airtable", "webflow",
    "make", "integromat", "notion", "figma", "zapier", "github", "actions", "cloud",
    "function", "docker", "nginx", "substack", "ghost"
])

def extract_keywords(text):
    """Extracts simple keywords from text, removing stop words."""
    words = re.findall(r'\b\w+\b', text.lower())
    return [word for word in words if word not in STOP_WORDS and len(word) > 2]

def get_promising_themes(top_n=5):
    """Analyzes high-rated ideas from DB and returns top keywords/themes."""
    # Note: This DB call is synchronous
    high_rated_ideas_data = state_manager.get_high_rated_ideas(limit=50)
    if not high_rated_ideas_data:
        return []

    logging.info(f"Analyzing {len(high_rated_ideas_data)} high-rated ideas for trends...")
    all_keywords = []
    for item in high_rated_ideas_data:
        all_keywords.extend(extract_keywords(item['name']))

    if not all_keywords:
        logging.warning("No meaningful keywords extracted from high-rated idea names.")
        return []

    keyword_counts = Counter(all_keywords)
    promising_keywords = [kw for kw, count in keyword_counts.most_common(top_n)]
    logging.info(f"Identified promising keywords/themes: {promising_keywords}")
    return promising_keywords

# --- Core Logic Functions ---

async def generate_ideas(session, full_prompt):
    """Generates a batch of SaaS ideas using the Gemini API via api_clients."""
    logging.info(f">>> Generating new SaaS ideas...")
    response_text = await api_clients.call_gemini_api_async(session, full_prompt)

    if not response_text:
        logging.warning("Failed to generate ideas from Gemini.")
        return []

    ideas = []
    for line in response_text.strip().split('\n'):
        line = line.strip()
        match = re.match(r"^\d+\.\s*(.*)", line)
        if match:
            idea_name = match.group(1).strip()
            if 0 < len(idea_name) < 200:
                ideas.append(idea_name)
            else:
                logging.warning(f"Ignoring potentially invalid idea name: '{idea_name}'")
        elif line:
            logging.warning(f"Ignoring unexpected line in idea generation response: '{line}'")

    if not ideas:
         logging.warning("Could not parse any valid ideas from Gemini response.")
    else:
        logging.info(f"Successfully parsed {len(ideas)} ideas.")
    return ideas

async def get_search_queries(session, idea_name):
    """Generates targeted search queries for an idea using Gemini via api_clients."""
    logging.info(f"--- Generating search queries for: '{idea_name}' ---")
    prompt = config.SEARCH_QUERY_GENERATION_PROMPT_TEMPLATE.format(idea_name=idea_name)
    response_text = await api_clients.call_gemini_api_async(session, prompt)
    if response_text:
        queries = [q.strip() for q in response_text.split('\n') if q.strip()]
        if len(queries) == 3:
            logging.info(f"Generated queries: {queries}")
            queries.append(f"{idea_name} review")
            return queries
        else:
            logging.warning(f"Gemini returned unexpected number of search queries ({len(queries)}). Using defaults.")
    else:
        logging.warning(f"Failed to generate search queries for '{idea_name}'. Using defaults.")

    return [
        f"{idea_name} alternatives",
        f"{idea_name} pricing",
        f"{idea_name} market need",
        f"{idea_name} review"
    ]

async def research_idea(session, idea_name):
    """Performs web searches asynchronously for a given idea and compiles a summary."""
    logging.info(f">>> Researching idea: '{idea_name}'...")
    queries = await get_search_queries(session, idea_name)
    research_summary = ""

    search_tasks = [api_clients.call_google_search_api_async(session, query) for query in queries]
    search_results_list = await asyncio.gather(*search_tasks, return_exceptions=True)

    for query, search_result_or_exc in zip(queries, search_results_list):
        if isinstance(search_result_or_exc, Exception):
            logging.error(f"Search task for query '{query}' failed: {search_result_or_exc}")
            continue

        search_results = search_result_or_exc
        if search_results and 'items' in search_results:
            research_summary += f"\nSearch Results for '{query}':\n"
            count = 0
            for result in search_results['items']:
                if count >= config.SEARCH_RESULTS_LIMIT:
                    break
                title = result.get('title', 'No Title')
                snippet = result.get('snippet', 'No Snippet')
                link = result.get('link', '#')
                entry = f"- {title}: {snippet} ({link})\n"
                if len(research_summary) + len(entry) < config.MAX_SUMMARY_LENGTH:
                     research_summary += entry
                     count += 1
                else:
                    logging.warning("Research summary truncated due to length limit.")
                    break
            research_summary += "-" * 10 + "\n"
        else:
            logging.warning(f"No organic results or error for query: '{query}'")

    logging.info(f"Research complete for '{idea_name}'. Summary length: {len(research_summary)}")
    return research_summary.strip()

async def rate_idea(session, idea_name, research_summary):
    """
    Rates an idea based on research using the Gemini API asynchronously.
    Parses individual criteria scores and justifications, calculates weighted average.
    Returns a tuple: (weighted_score, justifications_dict) or (None, None) on failure.
    """
    logging.info(f">>> Rating idea: '{idea_name}'...")
    if not research_summary:
        logging.warning("Skipping rating due to empty research summary.")
        return None, None

    prompt = config.RATING_PROMPT_TEMPLATE.format(
        idea_name=idea_name,
        research_summary=research_summary
    )
    response_text = await api_clients.call_gemini_api_async(session, prompt)

    if not response_text:
        logging.warning(f"Failed to get rating breakdown from Gemini for '{idea_name}'.")
        return None, None

    scores = {}
    justifications = {}
    expected_keys = ["need", "willingnesstopay", "competition", "monetization", "feasibility"]
    lines = response_text.strip().split('\n')
    parsed_count = 0
    score_pattern = re.compile(r"^\s*(\w+)\s*:\s*([\d\.]+)\s*\|\s*Justification:\s*(.*)$", re.IGNORECASE)

    for line in lines:
        match = score_pattern.match(line.strip())
        if match:
            key = match.group(1).lower()
            score_str = match.group(2)
            justification = match.group(3).strip()
            try:
                value = float(score_str)
                if 0.0 <= value <= 10.0:
                    found_key = next((ek for ek in expected_keys if ek == key), None)
                    if found_key:
                        scores[found_key] = value
                        justifications[found_key] = justification
                        parsed_count += 1
                    else:
                         logging.warning(f"Unexpected key '{key}' in rating response line: '{line}'")
                else:
                    logging.warning(f"Score out of range (0-10) in rating response line: '{line}'")
            except ValueError:
                 logging.warning(f"Could not convert score to float in rating line: '{line}'")
        elif line.strip():
            logging.warning(f"Could not parse rating line format: '{line}'")

    if parsed_count != len(expected_keys):
         missing = set(expected_keys) - set(scores.keys())
         logging.error(f"Parsed {parsed_count}/5 scores. Missing scores/justifications for criteria: {missing}. Response: '{response_text}'")
         return None, None

    weighted_total = 0
    for key, score in scores.items():
        weight = config.RATING_WEIGHTS.get(key, 1.0 / len(expected_keys))
        weighted_total += score * weight
    final_score = weighted_total

    logging.info(f"Received rating breakdown for '{idea_name}': {scores}. Weighted Score: {final_score:.1f}")
    return final_score, justifications


async def process_single_idea(idea_name, processed_ideas_set, session, semaphore):
    """Async function to research, rate, and potentially save/update state for a single idea."""
    async with semaphore: # Control concurrency
        idea_lower = idea_name.lower()
        if idea_lower in processed_ideas_set:
            logging.info(f"Skipping already processed idea (checked again): '{idea_name}'")
            return

        status = "processing"
        rating = None
        justifications = {}
        try:
            status = "researching"
            summary = await research_idea(session, idea_name)

            await asyncio.sleep(random.uniform(0.5, 1.5))

            status = "rating"
            rating, justifications = await rate_idea(session, idea_name, summary)

            if rating is not None:
                status = "rated"
                if rating >= config.RATING_THRESHOLD:
                    state_manager.save_rated_idea(idea_name, rating, justifications if justifications else {})
            else:
                status = "rating_failed"

            logging.info(f"--- Finished processing idea: '{idea_name}' ---")

        except Exception as e:
            logging.error(f"Unexpected error processing idea '{idea_name}': {e}", exc_info=True)
            status = "error"
        finally:
            state_manager.update_idea_state(idea_name, status, rating)
            processed_ideas_set.add(idea_lower)

            logging.info(f"--- Waiting {config.DELAY_BETWEEN_IDEAS} seconds before next idea ---")
            await asyncio.sleep(config.DELAY_BETWEEN_IDEAS)


# --- Main Execution ---

async def main():
    """Main asynchronous function to run the automator."""
    config.setup_logging()
    logging.info("Starting SaaS Idea Automator (Async, Modular, Enhanced, SQLite)...")

    try:
        state_manager.init_db()
    except Exception as e:
        logging.critical(f"Failed to initialize database. Exiting. Error: {e}")
        sys.exit(1)

    if not config.validate_config():
        logging.critical("API Keys or CSE ID missing in config/.env file. Exiting.")
        sys.exit(1)

    processed_ideas_set = state_manager.load_processed_ideas()

    if not os.path.exists(config.OUTPUT_FILE):
         try:
             with open(config.OUTPUT_FILE, 'w', encoding='utf-8') as f:
                  f.write("# AI-Rated SaaS Ideas (Weighted Score >= 7.5)\n\n")
                  f.write("Ideas are rated based on Need, WillingnessToPay, Competition, Monetization, Feasibility.\n")
                  f.write("Justifications are AI-generated based *only* on limited web search summaries.\n\n")
         except Exception as e:
             logging.error(f"Could not create output file '{config.OUTPUT_FILE}': {e}")
             return

    run_count = 0
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_TASKS)
    promising_themes = [] # Store promising themes across runs

    async with aiohttp.ClientSession() as session:
        while run_count < config.MAX_RUNS:
            run_count += 1
            batch_start_time = time.time() # Record start time for duration calculation
            logging.info(f"===== Starting Run {run_count}/{config.MAX_RUNS} =====")

            # --- Periodic Trend Analysis ---
            if run_count > 1 and (run_count - 1) % 5 == 0:
                promising_themes = get_promising_themes(top_n=5)

            # --- Determine Generation Strategy (Explore/Exploit) ---
            explore = random.random() < config.EXPLORE_RATIO
            generation_prompt = config.IDEA_GENERATION_PROMPT_TEMPLATE.format(num_ideas=config.IDEAS_PER_BATCH)

            # --- Negative Feedback ---
            num_avoid_examples = 10
            low_rated_ideas = state_manager.get_low_rated_ideas(threshold=4.0, limit=num_avoid_examples)
            if low_rated_ideas:
                logging.info(f"Adding {len(low_rated_ideas)} low-rated ideas to negative feedback.")
                avoid_prompt_part = "\n\nAvoid generating ideas conceptually similar to these low-rated examples:\n"
                for avoid_idea in low_rated_ideas:
                    avoid_prompt_part += f"- {avoid_idea}\n"
                generation_prompt += avoid_prompt_part
            # Add fallback using random processed ideas if needed (optional)

            # --- Positive Feedback (Exploit Phase) ---
            if not explore and promising_themes:
                logging.info(f"Exploiting promising themes: {promising_themes}")
                focus_prompt_part = "\n\nTry to generate ideas related to these promising themes if possible:\n"
                for theme in promising_themes:
                    focus_prompt_part += f"- {theme}\n"
                generation_prompt += focus_prompt_part
            elif explore:
                 logging.info("Exploring with broader prompt.")
            else:
                 logging.info("No promising themes yet, using standard prompt.")


            # --- Generate Ideas ---
            initial_ideas = await generate_ideas(session, generation_prompt)

            # --- Self-Critique Step ---
            ideas = []
            if initial_ideas:
                logging.info(f"Performing self-critique on {len(initial_ideas)} generated ideas...")
                idea_list_str = "\n".join([f"{i+1}. {idea}" for i, idea in enumerate(initial_ideas)])
                critique_prompt = config.SELF_CRITIQUE_PROMPT_TEMPLATE.format(idea_list_str=idea_list_str)
                critique_response = await api_clients.call_gemini_api_async(session, critique_prompt)

                if critique_response and critique_response.lower().strip() != "none":
                    critiqued_ideas_list = []
                    for line in critique_response.strip().split('\n'):
                        line = line.strip()
                        match = re.match(r"^\d+\.\s*(.*)", line)
                        if match:
                            idea_name = match.group(1).strip()
                            if 0 < len(idea_name) < 200:
                                critiqued_ideas_list.append(idea_name)
                        elif line:
                             logging.warning(f"Ignoring unexpected line in self-critique response: '{line}'")

                    original_ideas_lower = {i.lower() for i in initial_ideas}
                    ideas = [idea for idea in critiqued_ideas_list if idea.lower() in original_ideas_lower]
                    logging.info(f"Self-critique resulted in {len(ideas)} ideas passing filter.")
                elif critique_response and critique_response.lower().strip() == "none":
                    logging.info("Self-critique resulted in no ideas passing filter.")
                    ideas = []
                else:
                    logging.warning("Self-critique call failed or returned empty/invalid response. Proceeding with initial ideas.")
                    ideas = initial_ideas
            # --- End Self-Critique ---

            if not ideas:
                logging.warning("No valid ideas to process after generation/critique.")
                await asyncio.sleep(10)
                continue

            logging.info(f"Attempting to process {len(ideas)} filtered ideas.")
            tasks = []
            new_ideas_processed_in_batch = 0 # Counter for ping message
            for idea in ideas:
                 if idea.lower() not in processed_ideas_set:
                     # Pass the counter reference or use a shared object if needed,
                     # but simpler to just count tasks created for the ping message
                     task = asyncio.create_task(process_single_idea(idea, processed_ideas_set, session, semaphore))
                     tasks.append(task)
                     new_ideas_processed_in_batch += 1
                 else:
                     logging.info(f"Skipping idea (already processed, missed by initial filter?): '{idea}'")

            if not tasks:
                logging.warning("No new ideas to process in this batch after filtering processed.")
            else:
                logging.info(f"Created {len(tasks)} tasks for new ideas.")
                await asyncio.gather(*tasks)

            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time # Calculate duration
            logging.info(f"===== Finished Run {run_count}/{config.MAX_RUNS} in {batch_duration:.2f} seconds =====")

            # --- Uptime Kuma Ping ---
            # Send ping with status message and duration metric
            status_message = f"Finished Run {run_count}/{config.MAX_RUNS}. Processed {new_ideas_processed_in_batch} new ideas."
            await api_clients.ping_uptime_kuma(session, message=status_message, ping_value=int(batch_duration))
            # -----------------------

            if run_count < config.MAX_RUNS:
                 logging.info(f"--- Waiting {config.WAIT_BETWEEN_BATCHES} seconds before next batch ---")
                 await asyncio.sleep(config.WAIT_BETWEEN_BATCHES)

    logging.info("SaaS Idea Automator finished.")

if __name__ == "__main__":
    missing_libs = []
    try: import aiohttp
    except ImportError: missing_libs.append("aiohttp")
    try: import dotenv
    except ImportError: missing_libs.append("python-dotenv")
    try: import requests # Keep requests check as it might be used elsewhere or as fallback
    except ImportError: missing_libs.append("requests")
    try: import sqlite3 # Check for sqlite3 (standard library but good practice)
    except ImportError: missing_libs.append("sqlite3") # Should not happen
    try: from collections import Counter # Check if Counter is available
    except ImportError: missing_libs.append("collections.Counter") # Should not happen

    if missing_libs:
        print(f"Error: Missing required libraries: {', '.join(missing_libs)}")
        print(f"Please install them using: pip install aiohttp python-dotenv requests")
        sys.exit(1)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.warning("Script interrupted by user.")
    except Exception as e:
        logging.critical(f"Unhandled exception in main execution: {e}", exc_info=True)
        sys.exit(1)
