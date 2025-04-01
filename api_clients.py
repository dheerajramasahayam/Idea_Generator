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

            # --- Smarter Retry Delay for 429 ---
            delay = base_delay * (2 ** attempt) # Default exponential backoff
            if isinstance(e, aiohttp.ClientResponseError) and e.status == 429:
                try:
                    # Attempt to parse retryDelay from the error message/body
                    # Note: Accessing e.message might not always work depending on aiohttp version/error details
                    # A more robust way might involve reading the response body before raising status
                    error_body = await e.text() # Try reading body
                    error_data = json.loads(error_body)
                    if isinstance(error_data, dict) and 'error' in error_data and 'details' in error_data['error']:
                        for detail in error_data['error']['details']:
                             if isinstance(detail, dict) and detail.get('@type') == 'type.googleapis.com/google.rpc.RetryInfo':
                                 retry_delay_str = detail.get('retryDelay')
                                 if isinstance(retry_delay_str, str) and retry_delay_str.endswith('s'):
                                     try:
                                         delay_seconds = int(retry_delay_str[:-1])
                                         delay = max(delay_seconds, base_delay)
                                         logging.info(f"Using API suggested retry delay: {delay}s")
                                         break
                                     except ValueError:
                                         logging.warning(f"Could not parse seconds from retryDelay: {retry_delay_str}")
                except (json.JSONDecodeError, KeyError, IndexError, TypeError, AttributeError) as parse_error:
                     logging.warning(f"Could not parse RetryInfo from 429 error details: {parse_error}. Falling back to exponential backoff.")
            # --- End Smarter Retry Delay ---

            delay_with_jitter = delay + random.uniform(0, 1)
            logging.info(f"Retrying Gemini API in {delay_with_jitter:.2f} seconds...")
            await asyncio.sleep(delay_with_jitter)
    return None

async def call_search_api_async(session, query):
    """
    Calls the configured search API (Brave, Serper, or Google) asynchronously.
    Requires the corresponding API key to be set in config/.env.
    """
    provider = config.SEARCH_PROVIDER
    logging.info(f"--- Searching via {provider.upper()} API for: '{query}' ---")

    if provider == "brave":
        # --- Brave Search API Implementation ---
        if not config.BRAVE_API_KEY or config.BRAVE_API_KEY == "YOUR_BRAVE_API_KEY":
            logging.error("Brave API key missing or not configured.")
            return None
        # !!! ASSUMPTION: Endpoint and parameters - VERIFY WITH BRAVE DOCS !!!
        url = "https://api.search.brave.com/res/v1/web/search"
        params = {'q': query, 'count': config.SEARCH_RESULTS_LIMIT}
        headers = {
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip',
            'X-Subscription-Token': config.BRAVE_API_KEY
        }
        api_name = "Brave Search"

    elif provider == "serper":
        # --- Serper API Implementation ---
        if not config.SERPER_API_KEY or config.SERPER_API_KEY == "YOUR_SERPER_API_KEY":
            logging.error("Serper API key missing or not configured.")
            return None
        url = "https://google.serper.dev/search"
        headers = {'X-API-KEY': config.SERPER_API_KEY, 'Content-Type': 'application/json'}
        payload = json.dumps({'q': query, 'num': config.SEARCH_RESULTS_LIMIT})
        params = None # Serper uses payload
        api_name = "Serper"

    elif provider == "google":
        # --- Google Custom Search API Implementation ---
        if not config.GOOGLE_API_KEY or config.GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY" or \
           not config.GOOGLE_CSE_ID or config.GOOGLE_CSE_ID == "YOUR_GOOGLE_CSE_ID":
            logging.error("Google Search API key or CSE ID missing.")
            return None
        url = "https://www.googleapis.com/customsearch/v1"
        params = {'key': config.GOOGLE_API_KEY, 'cx': config.GOOGLE_CSE_ID, 'q': query, 'num': config.SEARCH_RESULTS_LIMIT}
        headers = None
        api_name = "Google Custom Search"

    else:
        logging.error("No valid search provider configured.")
        return None

    max_retries = 3
    base_delay = 1

    for attempt in range(max_retries):
        try:
            request_args = {'params': params} if params else {'data': payload}
            async with session.get(url, headers=headers, timeout=15, **request_args) as response:
                status = response.status
                if status == 429: # Rate limit
                    retry_after = int(response.headers.get("Retry-After", base_delay * (3 ** attempt)))
                    logging.warning(f"{api_name} API attempt {attempt + 1} failed: Rate limit (429). Retrying after {retry_after}s...")
                    await asyncio.sleep(retry_after + random.uniform(0, 1))
                    continue
                elif status >= 500: # Server errors
                    logging.warning(f"{api_name} API attempt {attempt + 1} failed with status {status}. Retrying...")
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(delay)
                    continue

                response.raise_for_status()
                return await response.json() # Return raw JSON, parsing happens in research_idea
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
            logging.error(f"{api_name} API attempt {attempt + 1} failed: {e}")
            if attempt + 1 == max_retries:
                logging.error(f"Max retries reached for {api_name} API.")
                return None
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            logging.info(f"Retrying {api_name} API in {delay:.2f} seconds...")
            await asyncio.sleep(delay)
    return None


async def ping_uptime_kuma(session, message="OK", ping_value=None):
    """Sends a heartbeat ping with optional message and ping value to Uptime Kuma."""
    base_url = config.UPTIME_KUMA_PUSH_URL
    if not base_url or base_url == "YOUR_PUSH_URL_HERE":
        logging.debug("Uptime Kuma Push URL not configured. Skipping ping.")
        return

    params = {"status": "up", "msg": message}
    if ping_value is not None:
        try:
            params["ping"] = int(ping_value)
        except (ValueError, TypeError):
            logging.warning(f"Could not convert ping_value '{ping_value}' to integer for Uptime Kuma.")

    try:
        # Use aiohttp's URL builder which handles None values in params correctly
        url = aiohttp.helpers.build_url(base_url, params)
        logging.debug(f"Pinging Uptime Kuma: {url}")
        async with session.get(str(url), timeout=10) as response: # Ensure URL is string
            if response.status != 200:
                 response_text = await response.text()
                 logging.warning(f"Uptime Kuma ping failed with status {response.status}. Response: {response_text[:200]}")
            else:
                 logging.info(f"Uptime Kuma ping successful. Message: '{message}', Ping: {params.get('ping', 'N/A')}")
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logging.error(f"Error pinging Uptime Kuma: {e}")
    except Exception as e: # Catch potential URL building errors too
         logging.error(f"Unexpected error during Uptime Kuma ping setup or execution: {e}")
