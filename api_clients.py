import aiohttp
import asyncio
import json
import random
import logging
import config # Import configuration

async def call_gemini_api_async(session, prompt):
    """
    Makes an async call to the Google Gemini API using aiohttp with improved retry.
    Uses configuration from config.py.
    """
    logging.info(f"--- Calling Google Gemini API ({config.GEMINI_MODEL}) ---")
    # logging.debug(f"Prompt:\n{prompt}") # Uncomment for full prompt logging

    if not config.GEMINI_API_KEY or config.GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        logging.error("Gemini API key is missing or not configured in .env")
        return None

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.GEMINI_MODEL}:generateContent?key={config.GEMINI_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]})
    max_retries = 3
    base_delay = 2 # Start with slightly longer base delay for Gemini

    for attempt in range(max_retries):
        try:
            async with session.post(api_url, headers=headers, data=payload, timeout=45) as response: # Longer timeout
                status = response.status
                if status == 429: # Rate limit
                    retry_after = int(response.headers.get("Retry-After", base_delay * (3 ** attempt))) # Use header if available, else longer exponential
                    logging.warning(f"Gemini API attempt {attempt + 1} failed: Rate limit (429). Retrying after {retry_after}s...")
                    await asyncio.sleep(retry_after + random.uniform(0, 1))
                    continue # Go to next attempt
                elif status >= 500: # Server errors
                    logging.warning(f"Gemini API attempt {attempt + 1} failed with status {status}. Retrying...")
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(delay)
                    continue # Go to next attempt

                response.raise_for_status() # Raise for other client errors (4xx)
                response_data = await response.json()

                # Parse response
                if 'candidates' in response_data and response_data['candidates']:
                    content = response_data['candidates'][0].get('content')
                    if content and 'parts' in content and content['parts']:
                        text_response = content['parts'][0].get('text')
                        if text_response:
                            logging.info("Gemini API call successful.")
                            return text_response.strip()

                # Handle blocks or unexpected structure
                block_reason = response_data.get('promptFeedback', {}).get('blockReason')
                if block_reason:
                    logging.warning(f"Gemini API blocked prompt. Reason: {block_reason}")
                    return None # Don't retry blocked prompts
                else:
                    logging.error(f"Unexpected Gemini response structure: {response_data}")
                    return None # Don't retry if structure is wrong

        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
            logging.error(f"Gemini API call attempt {attempt + 1} failed: {e}")
            if attempt + 1 == max_retries:
                logging.error("Max retries reached for Gemini API.")
                return None
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            logging.info(f"Retrying Gemini API in {delay:.2f} seconds...")
            await asyncio.sleep(delay)
    return None

async def call_google_search_api_async(session, query):
    """Calls the official Google Custom Search JSON API asynchronously with improved retry."""
    logging.info(f"--- Searching Google Search API for: '{query}' ---")

    if not config.GOOGLE_API_KEY or config.GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY" or \
       not config.GOOGLE_CSE_ID or config.GOOGLE_CSE_ID == "YOUR_GOOGLE_CSE_ID":
        logging.error("Google Search API key or CSE ID is missing or not configured in .env")
        return None

    url = "https://www.googleapis.com/customsearch/v1"
    params = {'key': config.GOOGLE_API_KEY, 'cx': config.GOOGLE_CSE_ID, 'q': query, 'num': config.SEARCH_RESULTS_LIMIT}
    max_retries = 3
    base_delay = 1

    for attempt in range(max_retries):
        try:
            async with session.get(url, params=params, timeout=15) as response:
                status = response.status
                if status == 429: # Rate limit
                    retry_after = int(response.headers.get("Retry-After", base_delay * (3 ** attempt)))
                    logging.warning(f"Google Search API attempt {attempt + 1} failed: Rate limit (429). Retrying after {retry_after}s...")
                    await asyncio.sleep(retry_after + random.uniform(0, 1))
                    continue
                elif status >= 500: # Server errors
                    logging.warning(f"Google Search API attempt {attempt + 1} failed with status {status}. Retrying...")
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(delay)
                    continue

                response.raise_for_status()
                return await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
            logging.error(f"Google Search API attempt {attempt + 1} failed: {e}")
            if attempt + 1 == max_retries:
                logging.error("Max retries reached for Google Search API.")
                return None
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            logging.info(f"Retrying Google Search API in {delay:.2f} seconds...")
            await asyncio.sleep(delay)
    return None

async def ping_uptime_kuma(session):
    """Sends a heartbeat ping to the configured Uptime Kuma Push URL."""
    if not config.UPTIME_KUMA_PUSH_URL or config.UPTIME_KUMA_PUSH_URL == "YOUR_PUSH_URL_HERE":
        logging.debug("Uptime Kuma Push URL not configured. Skipping ping.")
        return

    logging.debug(f"Pinging Uptime Kuma: {config.UPTIME_KUMA_PUSH_URL}")
    try:
        # Simple GET request, timeout is short as it should be fast
        async with session.get(config.UPTIME_KUMA_PUSH_URL, timeout=10) as response:
            # We generally don't need to check the response status for Kuma push,
            # but logging errors is good.
            if response.status != 200:
                 logging.warning(f"Uptime Kuma ping failed with status {response.status}")
            else:
                 logging.info("Uptime Kuma ping successful.")
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logging.error(f"Error pinging Uptime Kuma: {e}")
    # Do not retry pings aggressively, as the main loop will try again next batch.
