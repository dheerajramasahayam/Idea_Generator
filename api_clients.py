import aiohttp
import asyncio
import json
import random
import logging
import config # Import configuration
from yarl import URL # Import URL from yarl
import time # Import time for GitHub rate limit handling

async def call_gemini_api_async(session, prompt):
    """
    Makes an async call to the Google Gemini API using aiohttp with improved retry.
    Uses configuration from config.py.
    """
    logging.info(f"--- Calling Google Gemini API ({config.GEMINI_MODEL}) ---")
    # logging.debug(f"Prompt:\n{prompt}")

    if not config.GEMINI_API_KEY or config.GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        logging.error("Gemini API key is missing or not configured in .env")
        return None

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.GEMINI_MODEL}:generateContent?key={config.GEMINI_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]})
    max_retries = 3; base_delay = 2

    for attempt in range(max_retries):
        try:
            async with session.post(api_url, headers=headers, data=payload, timeout=45) as response:
                status = response.status
                if status == 429:
                    retry_after = int(response.headers.get("Retry-After", base_delay * (3 ** attempt)))
                    logging.warning(f"Gemini API attempt {attempt + 1} failed: Rate limit (429). Retrying after {retry_after}s...")
                    await asyncio.sleep(retry_after + random.uniform(0, 1)); continue
                elif status >= 500:
                    logging.warning(f"Gemini API attempt {attempt + 1} failed with status {status}. Retrying...")
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1); await asyncio.sleep(delay); continue

                response.raise_for_status()
                response_data = await response.json()

                if 'candidates' in response_data and response_data['candidates']:
                    content = response_data['candidates'][0].get('content')
                    if content and 'parts' in content and content['parts']:
                        text_response = content['parts'][0].get('text')
                        if text_response: logging.info("Gemini API call successful."); return text_response.strip()

                block_reason = response_data.get('promptFeedback', {}).get('blockReason')
                if block_reason: logging.warning(f"Gemini API blocked prompt. Reason: {block_reason}"); return None
                else: logging.error(f"Unexpected Gemini response structure: {response_data}"); return None

        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
            logging.error(f"Gemini API call attempt {attempt + 1} failed: {e}")
            if attempt + 1 == max_retries: logging.error("Max retries reached for Gemini API."); return None
            delay = base_delay * (2 ** attempt)
            if isinstance(e, aiohttp.ClientResponseError) and e.status == 429:
                try:
                    error_body = await e.text(); error_data = json.loads(error_body)
                    if isinstance(error_data, dict) and 'error' in error_data and 'details' in error_data['error']:
                        for detail in error_data['error']['details']:
                             if isinstance(detail, dict) and detail.get('@type') == 'type.googleapis.com/google.rpc.RetryInfo':
                                 retry_delay_str = detail.get('retryDelay')
                                 if isinstance(retry_delay_str, str) and retry_delay_str.endswith('s'):
                                     try: delay = max(int(retry_delay_str[:-1]), base_delay); logging.info(f"Using API suggested retry delay: {delay}s"); break
                                     except ValueError: logging.warning(f"Could not parse seconds from retryDelay: {retry_delay_str}")
                except (json.JSONDecodeError, KeyError, IndexError, TypeError, AttributeError) as parse_error: logging.warning(f"Could not parse RetryInfo from 429 error details: {parse_error}. Falling back.")
            delay_with_jitter = delay + random.uniform(0, 1)
            logging.info(f"Retrying Gemini API in {delay_with_jitter:.2f} seconds...")
            await asyncio.sleep(delay_with_jitter)
    return None

async def generate_variation_ideas_async(session, variation_prompt):
    """Generates variations of an idea using the Gemini API."""
    logging.info(">>> Generating variation ideas...")
    return await call_gemini_api_async(session, variation_prompt)

async def generate_regenerated_ideas_async(session, regeneration_prompt):
    """Generates alternative ideas based on feedback using the Gemini API."""
    logging.info(">>> Generating alternative ideas (focused re-generation)...")
    return await call_gemini_api_async(session, regeneration_prompt)


async def call_search_api_async(session, query):
    """Calls the configured search API (Brave, Serper, or Google) asynchronously."""
    provider = config.SEARCH_PROVIDER
    logging.info(f"--- Searching via {provider.upper()} API for: '{query}' ---")
    url, params, headers, payload, api_name = None, None, None, None, None

    if provider == "brave":
        if not config.BRAVE_API_KEY or config.BRAVE_API_KEY == "YOUR_BRAVE_API_KEY": logging.error("Brave API key missing."); return None
        url = "https://api.search.brave.com/res/v1/web/search"
        params = {'q': query, 'count': config.SEARCH_RESULTS_LIMIT, 'freshness': 'pm', 'result_filter': 'web', 'extra_snippets': 'true'}
        headers = {'Accept': 'application/json', 'Accept-Encoding': 'gzip', 'X-Subscription-Token': config.BRAVE_API_KEY}
        api_name = "Brave Search"
    elif provider == "serper":
        if not config.SERPER_API_KEY or config.SERPER_API_KEY == "YOUR_SERPER_API_KEY": logging.error("Serper API key missing."); return None
        url = "https://google.serper.dev/search"
        headers = {'X-API-KEY': config.SERPER_API_KEY, 'Content-Type': 'application/json'}
        payload = json.dumps({'q': query, 'num': config.SEARCH_RESULTS_LIMIT})
        api_name = "Serper"
    elif provider == "google":
        if not config.GOOGLE_API_KEY or not config.GOOGLE_CSE_ID: logging.error("Google Search API key or CSE ID missing."); return None
        url = "https://www.googleapis.com/customsearch/v1"
        params = {'key': config.GOOGLE_API_KEY, 'cx': config.GOOGLE_CSE_ID, 'q': query, 'num': config.SEARCH_RESULTS_LIMIT}
        api_name = "Google Custom Search"
    else: logging.error("No valid search provider configured."); return None

    max_retries = 3; base_delay = 1
    for attempt in range(max_retries):
        try:
            processed_params = {k: str(v) if isinstance(v, bool) else v for k, v in params.items()} if params else None
            request_args = {'params': processed_params} if processed_params else {'data': payload}
            async with session.get(url, headers=headers, timeout=15, **request_args) as response:
                status = response.status
                if status == 429:
                    retry_after = int(response.headers.get("Retry-After", base_delay * (3 ** attempt)))
                    logging.warning(f"{api_name} API attempt {attempt + 1} failed: Rate limit (429). Retrying after {retry_after}s...")
                    await asyncio.sleep(retry_after + random.uniform(0, 1)); continue
                elif status >= 500:
                    logging.warning(f"{api_name} API attempt {attempt + 1} failed with status {status}. Retrying...")
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1); await asyncio.sleep(delay); continue
                response.raise_for_status()
                return await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
            logging.error(f"{api_name} API attempt {attempt + 1} failed: {e}")
            if attempt + 1 == max_retries: logging.error(f"Max retries reached for {api_name} API."); return None
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            logging.info(f"Retrying {api_name} API in {delay:.2f} seconds...")
            await asyncio.sleep(delay)
    return None

async def call_github_search_api_async(session, query):
    """Searches GitHub repositories using the GitHub REST API."""
    logging.info(f"--- Searching GitHub Repositories for: '{query}' ---")
    url = "https://api.github.com/search/repositories"
    params = {'q': query, 'sort': 'stars', 'order': 'desc', 'per_page': 5}
    headers = {'Accept': 'application/vnd.github.v3+json'}
    if config.GITHUB_PAT:
        logging.debug("Using GitHub PAT for authentication.")
        headers['Authorization'] = f"Bearer {config.GITHUB_PAT}"
    else:
        logging.warning("No GitHub PAT found. API calls will be severely rate-limited.")

    max_retries = 2; base_delay = 3
    for attempt in range(max_retries):
        try:
            async with session.get(url, headers=headers, params=params, timeout=20) as response:
                status = response.status
                remaining = response.headers.get('X-RateLimit-Remaining')
                if remaining: logging.debug(f"GitHub API rate limit remaining: {remaining}")

                if status == 403 and 'rate limit exceeded' in (await response.text()).lower():
                    reset_time = int(response.headers.get('X-RateLimit-Reset', time.time() + base_delay * (3**attempt)))
                    wait_time = max(reset_time - time.time(), base_delay)
                    logging.warning(f"GitHub API attempt {attempt + 1} failed: Rate limit exceeded. Retrying after {wait_time:.0f}s...")
                    await asyncio.sleep(wait_time + random.uniform(0, 1)); continue
                elif status >= 500:
                     logging.warning(f"GitHub API attempt {attempt + 1} failed with status {status}. Retrying...")
                     delay = base_delay * (2 ** attempt) + random.uniform(0, 1); await asyncio.sleep(delay); continue

                response.raise_for_status()
                data = await response.json()
                logging.info(f"GitHub API search successful. Found {data.get('total_count', 0)} potential repositories.")
                return data

        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
            logging.error(f"GitHub API call attempt {attempt + 1} failed: {e}")
            if attempt + 1 == max_retries: logging.error("Max retries reached for GitHub API."); return None
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            logging.info(f"Retrying GitHub API in {delay:.2f} seconds...")
            await asyncio.sleep(delay)
    return None


async def ping_uptime_kuma(session, message="OK", ping_value=None):
    """Sends a heartbeat ping with optional message and ping value to Uptime Kuma."""
    base_url_str = config.UPTIME_KUMA_PUSH_URL
    if not base_url_str or base_url_str == "YOUR_PUSH_URL_HERE": return
    try:
        base_url = URL(base_url_str)
        params = {"status": "up", "msg": message}
        if ping_value is not None:
            try: params["ping"] = str(int(ping_value))
            except (ValueError, TypeError): logging.warning(f"Invalid ping_value '{ping_value}' for Uptime Kuma.")
        ping_url = base_url.with_query(params)
        logging.debug(f"Pinging Uptime Kuma: {ping_url}")
        async with session.get(ping_url, timeout=10) as response:
            if response.status != 200: logging.warning(f"Uptime Kuma ping failed: {response.status}")
            else: logging.info(f"Uptime Kuma ping successful. Msg: '{message}', Ping: {params.get('ping', 'N/A')}")
    except Exception as e: logging.error(f"Error pinging Uptime Kuma: {e}")
