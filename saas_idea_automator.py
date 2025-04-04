import asyncio
import aiohttp
import json
import time
import random
import logging
import re
import sys
import os
import traceback
import math # For softmax
import numpy as np # For softmax
import config
import api_clients
import state_manager
import analysis_utils
import notifications
import nltk
import sentence_transformers
# import numpy # Already imported via analysis_utils potentially
import sklearn

# --- Helper Functions ---

def get_prompt_type_identifier(template_content):
    """Gets the identifier ('general', 'platform', etc.) for a given prompt template string."""
    # This relies on the PROMPT_TYPE_MAP created in config.py during prompt loading
    return config.PROMPT_TYPE_MAP.get(template_content, "unknown")

def softmax(x, temperature=1.0):
    """Compute softmax values for each sets of scores in x."""
    if not isinstance(x, np.ndarray): x = np.array(x)
    if temperature == 0: # Handle zero temperature -> greedy selection
        probs = np.zeros_like(x)
        probs[np.argmax(x)] = 1.0
        return probs
    e_x = np.exp((x - np.max(x)) / temperature) # subtract max for numerical stability
    return e_x / e_x.sum(axis=0)

# --- Core Logic Functions ---

async def generate_ideas(session, full_prompt):
    """Generates a batch of SaaS idea names using the Gemini API."""
    logging.info(f">>> Generating new SaaS idea names...")
    response_text = await api_clients.call_gemini_api_async(session, full_prompt)
    if not response_text: return []
    ideas = []
    for line in response_text.strip().split('\n'):
        match = re.match(r"^\d+\.\s*(.*)", line.strip())
        if match:
            idea_name = match.group(1).strip()
            if 0 < len(idea_name) < 200: ideas.append(idea_name)
            else: logging.warning(f"Ignoring potentially invalid idea name: '{idea_name}'")
        elif line.strip(): logging.warning(f"Ignoring unexpected line: '{line.strip()}'")
    if not ideas: logging.warning("Could not parse any valid idea names from Gemini response.")
    else: logging.info(f"Successfully parsed {len(ideas)} initial idea names.")
    return ideas

async def generate_descriptions(session, idea_names):
    """Generates descriptions for a list of idea names using Gemini."""
    if not idea_names: return {}
    logging.info(f">>> Generating descriptions for {len(idea_names)} ideas...")
    tasks = [api_clients.call_gemini_api_async(session, config.IDEA_DESCRIPTION_PROMPT_TEMPLATE.format(idea_name=name)) for name in idea_names]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    descriptions = {}
    for name, result_or_exc in zip(idea_names, results):
        if isinstance(result_or_exc, Exception): logging.error(f"Failed to generate description for '{name}': {result_or_exc}"); descriptions[name] = None
        elif not result_or_exc: logging.warning(f"Empty description returned for '{name}'."); descriptions[name] = None
        else:
            desc = result_or_exc.strip().strip('"').strip("'")
            if 0 < len(desc) < 500: descriptions[name] = desc; logging.debug(f"Generated description for '{name}': {desc}")
            else: logging.warning(f"Ignoring potentially invalid description for '{name}': {desc[:100]}..."); descriptions[name] = None
    logging.info(f"Finished generating descriptions (successful/failed): {len([d for d in descriptions.values() if d is not None])}/{len(descriptions)}")
    return descriptions


async def get_search_queries(session, idea_name):
    """Generates targeted search queries for an idea using Gemini."""
    logging.info(f"--- Generating search queries for: '{idea_name}' ---")
    prompt = config.SEARCH_QUERY_GENERATION_PROMPT_TEMPLATE.format(idea_name=idea_name)
    response_text = await api_clients.call_gemini_api_async(session, prompt)
    if response_text:
        queries = [q.strip() for q in response_text.split('\n') if q.strip()]
        if len(queries) == 3: logging.info(f"Generated queries: {queries}"); queries.append(f"{idea_name} review"); return queries
    logging.warning(f"Failed to generate search queries for '{idea_name}'. Using defaults.")
    return [f"{idea_name} alternatives", f"{idea_name} pricing", f"{idea_name} market need", f"{idea_name} review"]

def _is_dev_tool_idea(idea_name, description):
    """Checks if an idea seems related to developer tools or GitHub."""
    text_to_check = (idea_name + " " + (description or "")).lower()
    dev_keywords = ["github", " api ", "developer tool", "library", " sdk ", "open source", " code ", " repo", " git "]
    return any(keyword in text_to_check for keyword in dev_keywords)

async def research_idea(session, idea_name, idea_description):
    """Performs web searches and optionally GitHub search, compiling a summary."""
    logging.info(f">>> Researching idea: '{idea_name}'...")
    queries = await get_search_queries(session, idea_name)
    research_summary = f"Idea: {idea_name}\nDescription: {idea_description or 'N/A'}\n"

    # --- Web Search ---
    search_tasks = [api_clients.call_search_api_async(session, query) for query in queries]
    search_results_list = await asyncio.gather(*search_tasks, return_exceptions=True)
    for query, result_or_exc in zip(queries, search_results_list):
        if isinstance(result_or_exc, Exception): logging.error(f"Search task failed for '{query}': {result_or_exc}"); continue
        search_results = result_or_exc
        if not search_results: continue
        research_summary += f"\nWeb Search Results for '{query}':\n"; count = 0; results_list = []
        provider = config.SEARCH_PROVIDER
        try:
            if provider == "brave" and 'web' in search_results and 'results' in search_results['web']:
                 results_list, keys = search_results['web']['results'], ('title', 'description', 'url')
                 if config.BRAVE_API_KEY and config.BRAVE_API_KEY != "YOUR_BRAVE_API_KEY":
                     extra_snippet_key = 'extra_snippets'
                     for result in results_list:
                         if count >= config.SEARCH_RESULTS_LIMIT: break
                         title, snippet, link = result.get(keys[0], 'N/A'), result.get(keys[1], 'N/A'), result.get(keys[2], '#')
                         entry = f"- {title}: {snippet} ({link})\n"
                         extras = result.get(extra_snippet_key, [])
                         if extras and isinstance(extras, list):
                              entry += "".join([f"  - Snippet: {extra}\n" for extra in extras])
                         if len(research_summary) + len(entry) < config.MAX_SUMMARY_LENGTH: research_summary += entry; count += 1
                         else: logging.warning("Research summary truncated (web search)."); break
                     research_summary += "-" * 10 + "\n"
                     continue

            if provider == "google" and 'items' in search_results: results_list, keys = search_results['items'], ('title', 'snippet', 'link')
            elif provider == "serper" and 'organic' in search_results: results_list, keys = search_results['organic'], ('title', 'snippet', 'link')
            elif provider == "brave" and 'web' in search_results and 'results' in search_results['web']: results_list, keys = search_results['web']['results'], ('title', 'description', 'url')
            else: logging.warning(f"Unexpected search structure from {provider} for '{query}'."); continue

            for result in results_list:
                if count >= config.SEARCH_RESULTS_LIMIT: break
                title, snippet, link = result.get(keys[0], 'N/A'), result.get(keys[1], 'N/A'), result.get(keys[2], '#')
                entry = f"- {title}: {snippet} ({link})\n"
                if len(research_summary) + len(entry) < config.MAX_SUMMARY_LENGTH: research_summary += entry; count += 1
                else: logging.warning("Research summary truncated (web search)."); break
            research_summary += "-" * 10 + "\n"
        except Exception as parse_exc: logging.error(f"Error parsing search results for '{query}' from {provider}: {parse_exc}", exc_info=True)

    # --- GitHub Search (Conditional) ---
    if _is_dev_tool_idea(idea_name, idea_description):
        logging.info(f"Idea '{idea_name}' seems dev-related, performing GitHub search...")
        github_query = " ".join(analysis_utils.tokenize_and_clean(idea_name + " " + (idea_description or "")))
        if github_query:
            github_results = await api_clients.call_github_search_api_async(session, github_query)
            if github_results and 'items' in github_results:
                gh_summary = "\nGitHub Repository Search Results:\n"
                gh_count = 0
                for item in github_results['items']:
                    if gh_count >= 5: break
                    name = item.get('full_name', 'N/A')
                    desc = item.get('description', 'N/A')
                    url = item.get('html_url', '#')
                    stars = item.get('stargazers_count', 0)
                    lang = item.get('language', 'N/A')
                    entry = f"- [{name}]({url}): {desc} (Stars: {stars}, Lang: {lang})\n"
                    if len(research_summary) + len(gh_summary) + len(entry) < config.MAX_SUMMARY_LENGTH:
                        gh_summary += entry; gh_count += 1
                    else: logging.warning("Research summary truncated (GitHub search)."); break
                if gh_count > 0: research_summary += gh_summary + "-" * 10 + "\n"
            else: logging.info("No relevant GitHub repositories found or API failed.")
        else: logging.warning("Could not generate meaningful query for GitHub search.")

    logging.info(f"Research complete for '{idea_name}'. Final summary length: {len(research_summary)}")
    return research_summary.strip()

async def extract_facts_for_rating(session, idea_name, research_summary):
    """Uses Gemini to extract key facts relevant to rating criteria."""
    logging.info(f">>> Extracting facts for: '{idea_name}'...")
    if not research_summary: return None
    prompt = config.FACT_EXTRACTION_PROMPT_TEMPLATE.format(idea_name=idea_name, research_summary=research_summary)
    extracted_facts = await api_clients.call_gemini_api_async(session, prompt)
    if not extracted_facts: logging.warning(f"Fact extraction failed for '{idea_name}'."); return None
    logging.info(f"Successfully extracted facts for '{idea_name}'."); logging.debug(f"Extracted Facts:\n{extracted_facts}")
    return extracted_facts

async def rate_idea(session, idea_name, rating_context):
    """Rates an idea based on provided context (extracted facts or raw summary) using Gemini."""
    logging.info(f">>> Rating idea: '{idea_name}' using provided context...")
    if not rating_context: return None, None, None
    prompt = config.RATING_PROMPT_TEMPLATE.format(idea_name=idea_name, rating_context=rating_context)
    response_text = await api_clients.call_gemini_api_async(session, prompt)
    if not response_text: return None, None, None
    scores, justifications = {}, {}; expected_keys = ["need", "willingnesstopay", "competition", "monetization", "feasibility"]; parsed_count = 0
    score_pattern = re.compile(r"^\s*(\w+)\s*:\s*([\d\.]+)\s*\|\s*Justification:\s*(.*)$", re.IGNORECASE)
    for line in response_text.strip().split('\n'):
        match = score_pattern.match(line.strip())
        if match:
            key, score_str, justification = match.group(1).lower(), match.group(2), match.group(3).strip()
            try:
                value = float(score_str)
                if 0.0 <= value <= 10.0:
                    found_key = next((ek for ek in expected_keys if ek == key), None)
                    if found_key: scores[found_key], justifications[found_key] = value, justification; parsed_count += 1
                    else: logging.warning(f"Unexpected key '{key}' in rating line: '{line.strip()}'")
                else: logging.warning(f"Score out of range in rating line: '{line.strip()}'")
            except ValueError: logging.warning(f"Could not convert score in rating line: '{line.strip()}'")
        elif line.strip(): logging.warning(f"Could not parse rating line format: '{line.strip()}'")
    if parsed_count != len(expected_keys): logging.error(f"Parsed {parsed_count}/5 scores. Missing: {set(expected_keys) - set(scores.keys())}. Resp: '{response_text}'"); return None, None, None
    weighted_total = sum(score * config.RATING_WEIGHTS.get(key, 1.0 / len(expected_keys)) for key, score in scores.items())
    logging.info(f"Received rating breakdown for '{idea_name}': {scores}. Weighted Score: {weighted_total:.1f}")
    return weighted_total, justifications, scores

async def process_single_idea(idea_name, idea_description, idea_embedding, source_prompt_type, processed_ideas_set, session, semaphore, batch_stats):
    """Async function to research, extract facts, rate, save state, update performance, and potentially queue re-generation."""
    async with semaphore:
        idea_lower = idea_name.lower()
        if idea_lower in processed_ideas_set: return
        status, rating, justifications, scores = "processing", None, {}, None
        try:
            status = "researching"; summary = await research_idea(session, idea_name, idea_description)
            await asyncio.sleep(random.uniform(0.5, 1.0))
            status = "fact_extraction"; extracted_facts = await extract_facts_for_rating(session, idea_name, summary)
            rating_context = extracted_facts if extracted_facts else summary
            if not rating_context: raise ValueError("Empty rating context.")
            await asyncio.sleep(random.uniform(0.5, 1.0))
            status = "rating"; rating, justifications, scores = await rate_idea(session, idea_name, rating_context)
            if rating is not None:
                status = "rated"
                # Update prompt performance regardless of saving status
                if source_prompt_type:
                    state_manager.update_prompt_performance(source_prompt_type, rating)

                if rating >= config.RATING_THRESHOLD:
                    state_manager.save_rated_idea(idea_name, rating, justifications if justifications else {})
                    batch_stats['saved_ideas'].append(f"{idea_name} ({rating:.1f})"); status = "saved"
                elif config.ENABLE_FOCUSED_REGENERATION and rating < config.REGENERATION_TRIGGER_THRESHOLD:
                    logging.info(f"Idea '{idea_name}' scored {rating:.1f} (< {config.REGENERATION_TRIGGER_THRESHOLD}). Triggering focused re-generation.")
                    if scores and justifications:
                        weakest_criterion = min(scores, key=scores.get)
                        weakness_justification = justifications.get(weakest_criterion, "N/A")
                        logging.info(f"Weakest criterion: {weakest_criterion} ({scores[weakest_criterion]:.1f}). Justification: {weakness_justification}")
                        regen_prompt = config.IDEA_REGENERATION_PROMPT_TEMPLATE.format(
                            original_idea_name=idea_name, original_description=idea_description or "N/A",
                            weakest_criterion_name=weakest_criterion, weakness_justification=weakness_justification,
                            num_alternatives=config.NUM_REGENERATION_ATTEMPTS)
                        regen_response = await api_clients.generate_regenerated_ideas_async(session, regen_prompt)
                        if regen_response:
                             regen_ideas = [match.group(1).strip() for line in regen_response.strip().split('\n') if (match := re.match(r"^\d+\.\s*(.*)", line.strip()))]
                             logging.info(f"Focused Re-generation suggested alternatives for '{idea_name}': {regen_ideas}")
                             state_manager.add_pending_ideas(regen_ideas)
                        else: logging.warning(f"Focused re-generation failed for '{idea_name}'.")
                    else: logging.warning(f"Cannot perform focused re-generation for '{idea_name}': missing scores or justifications.")
            else: status = "rating_failed"
            logging.info(f"--- Finished processing idea: '{idea_name}' ---")
        except Exception as e: logging.error(f"Unexpected error processing idea '{idea_name}': {e}", exc_info=True); status = "error"; batch_stats['errors'] += 1
        finally:
            state_manager.update_idea_state(idea_name, status, rating, justifications, embedding=idea_embedding, description=idea_description, source_prompt_type=source_prompt_type)
            processed_ideas_set.add(idea_lower); batch_stats['processed'] += 1

# --- Main Execution ---
async def main():
    config.setup_logging(); logging.info("Starting SaaS Idea Automator...")
    try: state_manager.init_db()
    except Exception as e: logging.critical(f"DB Init Failed: {e}"); sys.exit(1)
    if not config.validate_config(): logging.critical("Config Invalid. Exiting."); sys.exit(1)
    if analysis_utils.embedding_model is None: logging.critical("Embedding model failed to load. Exiting."); sys.exit(1)

    processed_ideas_set = state_manager.load_processed_ideas()
    if not os.path.exists(config.OUTPUT_FILE):
         try:
             with open(config.OUTPUT_FILE, 'w', encoding='utf-8') as f: f.write(f"# AI-Rated SaaS Ideas (Score >= {config.RATING_THRESHOLD})\n\n[...Header...]\n\n")
         except Exception as e: logging.error(f"Could not create output file '{config.OUTPUT_FILE}': {e}"); return

    run_count, semaphore = 0, asyncio.Semaphore(config.MAX_CONCURRENT_TASKS)
    promising_themes, current_explore_ratio = [], config.EXPLORE_RATIO
    recent_ratings_window, adjustment_step = 50, 0.02
    min_explore, max_explore = 0.05, 0.50
    target_success_rate_upper, target_success_rate_lower = 0.15, 0.05

    async with aiohttp.ClientSession() as session:
        while run_count < config.MAX_RUNS:
            run_count += 1; batch_start_time = time.time()
            logging.info(f"===== Starting Run {run_count}/{config.MAX_RUNS} =====")
            batch_stats = {'processed': 0, 'saved_ideas': [], 'errors': 0}
            current_batch_prompt_types = {} # Track prompt type for ideas in this batch

            # --- Adaptive Explore/Exploit & Trend Analysis ---
            if run_count > 1:
                # ... (adaptive logic) ...
                recent_ratings = state_manager.get_recent_ratings(limit=recent_ratings_window)
                if len(recent_ratings) >= 10:
                    success_rate = sum(1 for r in recent_ratings if r >= config.RATING_THRESHOLD) / len(recent_ratings)
                    logging.info(f"Recent success rate ({len(recent_ratings)} ideas): {success_rate:.2%}")
                    if success_rate > target_success_rate_upper: current_explore_ratio = max(min_explore, current_explore_ratio - adjustment_step); logging.info(f"Decreasing explore ratio to {current_explore_ratio:.2f}")
                    elif success_rate < target_success_rate_lower: current_explore_ratio = min(max_explore, current_explore_ratio + adjustment_step); logging.info(f"Increasing explore ratio to {current_explore_ratio:.2f}")
                    else: logging.info(f"Keeping explore ratio at {current_explore_ratio:.2f}")
                else: logging.info(f"Not enough recent ratings ({len(recent_ratings)}) to adjust explore ratio.")

                if (run_count - 1) % config.TREND_ANALYSIS_RUN_INTERVAL == 0:
                     high_rated_ideas_data = state_manager.get_high_rated_ideas(threshold=config.RATING_THRESHOLD, limit=100)
                     if high_rated_ideas_data and len(high_rated_ideas_data) >= config.TREND_ANALYSIS_MIN_IDEAS:
                          promising_themes = analysis_utils.get_combined_themes(high_rated_ideas_data)
                     else: logging.info("Skipping trend analysis: Not enough high-rated ideas meeting threshold."); promising_themes = []
            else: logging.info(f"Initial explore ratio: {current_explore_ratio:.2f}"); promising_themes = []

            # --- Prepare Negative Feedback ---
            avoid_keywords_str = ""; avoid_examples_str = ""
            low_rated_texts = state_manager.get_low_rated_texts(limit=50)
            if low_rated_texts:
                 avoid_keywords = analysis_utils.extract_keywords(low_rated_texts, top_n=15)
                 if avoid_keywords: avoid_keywords_str = "\n\nAvoid ideas related to these keywords: " + ", ".join(avoid_keywords)
                 avoid_examples = random.sample(low_rated_texts, min(len(low_rated_texts), 3))
                 if avoid_examples: avoid_examples_str = "\n\nAlso avoid generating ideas conceptually similar to these low-rated examples:\n" + "".join([f"- {ex}\n" for ex in avoid_examples])

            # --- Idea Generation ---
            initial_ideas = []; generation_mode = "New Ideas (Default)"; source_prompt_type_for_batch = None
            pending_ideas = state_manager.fetch_and_clear_pending_ideas(limit=config.IDEAS_PER_BATCH)
            if pending_ideas:
                initial_ideas = pending_ideas; generation_mode = "Pending Re-generated"
                logging.info(f"Processing {len(initial_ideas)} pending re-generated ideas from queue.")
                # Note: We don't know the original source prompt type for re-generated ideas easily here.
                # Performance tracking will rely on the original idea's source type recorded during its processing.
            else: # No pending ideas, generate new ones
                if config.ENABLE_MULTI_STEP_GENERATION:
                    # ... (multi-step logic) ...
                    logging.info("Attempting multi-step generation strategy...")
                    generation_mode = "Multi-Step"
                    concepts_prompt = config.CONCEPT_GENERATION_PROMPT_TEMPLATE.format(num_concepts=config.NUM_CONCEPTS_TO_GENERATE)
                    concepts_text = await api_clients.call_gemini_api_async(session, concepts_prompt)
                    if concepts_text:
                        concepts = [match.group(1).strip() for line in concepts_text.strip().split('\n') if (match := re.match(r"^\d+\.\s*(.*)", line.strip()))]
                        if concepts:
                            selection_prompt = config.CONCEPT_SELECTION_PROMPT_TEMPLATE.format(num_to_select=config.NUM_CONCEPTS_TO_SELECT, concept_list_str="\n".join([f"{i+1}. {c}" for i, c in enumerate(concepts)]))
                            selected_concepts_text = await api_clients.call_gemini_api_async(session, selection_prompt)
                            if selected_concepts_text:
                                selected_concept_names = [match.group(1).strip() for line in selected_concepts_text.strip().split('\n') if (match := re.match(r"^\d+\.\s*(.*)", line.strip()))]
                                valid_selected_concepts = [c for c in selected_concept_names if c in concepts]
                                if valid_selected_concepts:
                                    specific_idea_tasks = []
                                    for concept in valid_selected_concepts:
                                        specific_prompt = config.SPECIFIC_IDEA_GENERATION_PROMPT_TEMPLATE.format(
                                                selected_concept=concept, num_ideas=config.NUM_IDEAS_PER_CONCEPT,
                                                avoid_keywords_section=avoid_keywords_str, avoid_examples_section=avoid_examples_str)
                                        specific_idea_tasks.append(api_clients.call_gemini_api_async(session, specific_prompt))
                                    specific_idea_results = await asyncio.gather(*specific_idea_tasks, return_exceptions=True)
                                    for result_or_exc in specific_idea_results:
                                        if isinstance(result_or_exc, Exception): logging.error(f"Specific idea generation failed: {result_or_exc}")
                                        elif result_or_exc:
                                            parsed_ideas = [match.group(1).strip() for line in result_or_exc.strip().split('\n') if (match := re.match(r"^\d+\.\s*(.*)", line.strip())) and 0 < len(match.group(1).strip()) < 200]
                                            initial_ideas.extend(parsed_ideas)
                                            # Associate these ideas with a generic 'multi-step' type for now
                                            for idea in parsed_ideas: current_batch_prompt_types[idea] = "multi-step"
                                else: logging.warning("Concept selection failed.")
                            else: logging.warning("Concept selection call failed.")
                        else: logging.warning("Failed to parse concepts.")
                    else: logging.warning("Concept generation call failed.")
                    if not initial_ideas: logging.warning("Multi-step generation failed. Falling back."); generation_mode = "Fallback"

                if not initial_ideas: # Fallback or Variation/New generation
                    if config.ENABLE_VARIATION_GENERATION and random.random() < config.VARIATION_GENERATION_PROBABILITY:
                        # ... (variation logic) ...
                        logging.info("Attempting variation generation strategy...")
                        generation_mode = "Variations"
                        candidate_ideas = state_manager.get_variation_candidate_ideas(config.VARIATION_SOURCE_MIN_RATING, config.VARIATION_SOURCE_MAX_RATING, limit=5)
                        if candidate_ideas:
                            # ... (variation prompt formatting) ...
                            source_idea = random.choice(candidate_ideas)
                            logging.info(f"Selected idea for variation: '{source_idea['name']}' (Rating: {source_idea['rating']})")
                            justifications = source_idea.get('justifications', {})
                            variation_prompt = config.IDEA_VARIATION_PROMPT_TEMPLATE.format(
                                original_idea_name=source_idea['name'], original_description=source_idea.get('description', 'N/A'),
                                need_score=justifications.get('need', 'N/A|N/A').split('|')[0].split(':')[-1].strip(), need_justification=justifications.get('need', 'N/A|N/A').split('|')[-1].split(':')[-1].strip(),
                                wtp_score=justifications.get('willingnesstopay', 'N/A|N/A').split('|')[0].split(':')[-1].strip(), wtp_justification=justifications.get('willingnesstopay', 'N/A|N/A').split('|')[-1].split(':')[-1].strip(),
                                comp_score=justifications.get('competition', 'N/A|N/A').split('|')[0].split(':')[-1].strip(), comp_justification=justifications.get('competition', 'N/A|N/A').split('|')[-1].split(':')[-1].strip(),
                                mon_score=justifications.get('monetization', 'N/A|N/A').split('|')[0].split(':')[-1].strip(), mon_justification=justifications.get('monetization', 'N/A|N/A').split('|')[-1].split(':')[-1].strip(),
                                feas_score=justifications.get('feasibility', 'N/A|N/A').split('|')[0].split(':')[-1].strip(), feas_justification=justifications.get('feasibility', 'N/A|N/A').split('|')[-1].split(':')[-1].strip(),
                                num_variations=config.NUM_VARIATIONS_TO_GENERATE)

                            variation_response = await api_clients.generate_variation_ideas_async(session, variation_prompt)
                            if variation_response:
                                parsed_variations = [match.group(1).strip() for line in variation_response.strip().split('\n') if (match := re.match(r"^\d+\.\s*(.*)", line.strip()))]
                                initial_ideas = [i for i in parsed_variations if 0 < len(i) < 200]
                                logging.info(f"Generated {len(initial_ideas)} variations.")
                                # Associate these ideas with 'variation' type
                                for idea in initial_ideas: current_batch_prompt_types[idea] = "variation"
                            else: logging.warning("Variation generation failed. Falling back.")
                        else: logging.info("No suitable candidates for variation. Generating new ideas.")

                    if not initial_ideas: # Fallback to standard new idea generation
                        generation_mode = "New Ideas"
                        explore = random.random() < current_explore_ratio
                        # --- Dynamic Prompt Selection Logic ---
                        if config.ENABLE_DYNAMIC_PROMPT_SELECTION:
                            logging.info("Attempting dynamic prompt selection...")
                            perf_data = state_manager.get_prompt_performance()
                            available_templates = config.IDEA_GENERATION_PROMPT_TEMPLATES
                            template_types = [get_prompt_type_identifier(t) for t in available_templates]
                            scores = []
                            default_score = 5.0 # Assign a neutral score for exploration if not enough data
                            for p_type in template_types:
                                data = perf_data.get(p_type)
                                if data and data['count'] >= config.DYNAMIC_SELECTION_MIN_DATA:
                                    scores.append(data['avg_rating'])
                                else:
                                    scores.append(default_score) # Encourage exploration
                            if not scores: # Should not happen if fallback exists
                                 base_prompt_template = random.choice(available_templates)
                            else:
                                 probabilities = softmax(scores, temperature=config.DYNAMIC_SELECTION_TEMP)
                                 logging.debug(f"Prompt types: {template_types}, Scores: {scores}, Probs: {probabilities}")
                                 base_prompt_template = random.choices(available_templates, weights=probabilities, k=1)[0]
                            source_prompt_type_for_batch = get_prompt_type_identifier(base_prompt_template)
                            logging.info(f"Dynamically selected prompt type: '{source_prompt_type_for_batch}'")
                        else: # Standard random selection
                            base_prompt_template = random.choice(config.IDEA_GENERATION_PROMPT_TEMPLATES)
                            source_prompt_type_for_batch = get_prompt_type_identifier(base_prompt_template)
                        # --- End Dynamic Selection ---

                        logging.info(f"Using generation prompt type: {source_prompt_type_for_batch}")
                        positive_themes_str = ""
                        if not explore and promising_themes:
                            logging.info(f"Exploiting promising themes: {promising_themes}")
                            positive_themes_str = "\n\nTry to generate ideas related to these concepts: " + ", ".join([f"'{t}'" for t in promising_themes])
                        elif explore: logging.info("Exploring with broader prompt.")
                        else: logging.info("No promising themes yet or exploring.")
                        generation_prompt = base_prompt_template.format(num_ideas=config.IDEAS_PER_BATCH, avoid_keywords_section=avoid_keywords_str, avoid_examples_section=avoid_examples_str)
                        generation_prompt += positive_themes_str
                        initial_ideas = await generate_ideas(session, generation_prompt)
                        # Associate these ideas with the chosen source type
                        for idea in initial_ideas: current_batch_prompt_types[idea] = source_prompt_type_for_batch


            # --- Generate Descriptions ---
            if not initial_ideas: logging.warning("No initial ideas generated."); await asyncio.sleep(10); continue
            idea_descriptions = await generate_descriptions(session, initial_ideas)

            # --- Semantic Negative Feedback Filter ---
            # ... (logic remains the same) ...
            ideas_after_neg_filter = []
            candidate_embeddings_dict = {}
            texts_to_embed = [idea_descriptions.get(name) for name in initial_ideas if idea_descriptions.get(name)]
            names_with_desc = [name for name in initial_ideas if idea_descriptions.get(name)]
            low_rated_embeddings = state_manager.get_low_rated_embeddings(limit=200)
            if low_rated_embeddings and texts_to_embed:
                logging.info("Applying semantic negative feedback filter based on descriptions...")
                candidate_embeddings_list = analysis_utils.generate_embeddings(texts_to_embed)
                if candidate_embeddings_list and len(candidate_embeddings_list) == len(texts_to_embed):
                    temp_embeddings_dict = {name: emb for name, emb in zip(names_with_desc, candidate_embeddings_list)}
                    for idea_name in initial_ideas:
                        embedding = temp_embeddings_dict.get(idea_name); description = idea_descriptions.get(idea_name)
                        if embedding and description:
                            if not analysis_utils.check_similarity(embedding, low_rated_embeddings, config.NEGATIVE_FEEDBACK_SIMILARITY_THRESHOLD):
                                ideas_after_neg_filter.append(idea_name); candidate_embeddings_dict[idea_name] = embedding
                            else: logging.info(f"Filtered out '{idea_name}' due to description similarity.")
                        elif description: ideas_after_neg_filter.append(idea_name); candidate_embeddings_dict[idea_name] = None
                else: logging.warning("Could not generate embeddings. Skipping semantic filter."); ideas_after_neg_filter = initial_ideas; candidate_embeddings_dict = {name: None for name in initial_ideas}
            else: logging.info("No low-rated embeddings or no valid descriptions. Skipping semantic filter."); ideas_after_neg_filter = initial_ideas; candidate_embeddings_dict = {name: None for name in initial_ideas}


            # --- Self-Critique Step ---
            # ... (logic remains the same) ...
            ideas = []
            if ideas_after_neg_filter:
                logging.info(f"Performing self-critique on {len(ideas_after_neg_filter)} ideas...")
                idea_list_str = "\n".join([f"{i+1}. {idea}" for i, idea in enumerate(ideas_after_neg_filter)])
                critique_prompt = config.SELF_CRITIQUE_PROMPT_TEMPLATE.format(idea_list_str=idea_list_str)
                critique_response = await api_clients.call_gemini_api_async(session, critique_prompt)
                if critique_response and critique_response.lower().strip() != "none":
                    critiqued_ideas_list = [match.group(1).strip() for line in critique_response.strip().split('\n') if (match := re.match(r"^\d+\.\s*(.*)", line.strip())) and 0 < len(match.group(1).strip()) < 200]
                    original_ideas_lower = {i.lower() for i in ideas_after_neg_filter}
                    ideas = [idea for idea in critiqued_ideas_list if idea.lower() in original_ideas_lower]
                    logging.info(f"Self-critique resulted in {len(ideas)} ideas passing filter.")
                elif critique_response and critique_response.lower().strip() == "none": logging.info("Self-critique resulted in no ideas passing filter.")
                else: logging.warning("Self-critique failed/empty. Using pre-critique list."); ideas = ideas_after_neg_filter


            # --- Process Final Ideas ---
            if not ideas: logging.warning("No ideas to process."); await asyncio.sleep(10); continue
            logging.info(f"Attempting to process {len(ideas)} final ideas.")
            # ... (embedding generation) ...
            final_idea_embeddings_map = {}; ideas_to_embed_final = []; final_descriptions_map = {}
            for idea in ideas:
                 final_descriptions_map[idea] = idea_descriptions.get(idea)
                 if idea in candidate_embeddings_dict and candidate_embeddings_dict[idea] is not None: final_idea_embeddings_map[idea] = candidate_embeddings_dict[idea]
                 elif final_descriptions_map[idea]: ideas_to_embed_final.append(final_descriptions_map[idea])
                 else: final_idea_embeddings_map[idea] = None
            if ideas_to_embed_final:
                 logging.info(f"Generating embeddings for {len(ideas_to_embed_final)} final idea descriptions...")
                 new_embeddings = analysis_utils.generate_embeddings(ideas_to_embed_final)
                 names_needing_embedding = [idea for idea in ideas if idea not in final_idea_embeddings_map]
                 if new_embeddings and len(new_embeddings) == len(names_needing_embedding):
                      for name, embedding in zip(names_needing_embedding, new_embeddings): final_idea_embeddings_map[name] = embedding
                 else:
                      logging.error("Failed to generate embeddings for some final ideas.");
                      for name in names_needing_embedding: final_idea_embeddings_map[name] = None


            # Pass the source prompt type (if known) to process_single_idea
            tasks = [asyncio.create_task(process_single_idea(
                        idea,
                        final_descriptions_map.get(idea),
                        final_idea_embeddings_map.get(idea),
                        current_batch_prompt_types.get(idea), # Pass type if available
                        processed_ideas_set, session, semaphore, batch_stats
                     )) for idea in ideas if idea.lower() not in processed_ideas_set]

            if not tasks: logging.warning("No new ideas to process after filtering processed.");
            else: logging.info(f"Created {len(tasks)} tasks for new ideas."); await asyncio.gather(*tasks)

            # --- Finish Run & Notify ---
            batch_duration = time.time() - batch_start_time
            logging.info(f"===== Finished Run {run_count}/{config.MAX_RUNS} in {batch_duration:.2f} seconds =====")
            # ... (notification logic) ...
            send_email_condition = config.ENABLE_EMAIL_NOTIFICATIONS and (len(batch_stats['saved_ideas']) > 0 or batch_stats['errors'] > 0)
            logging.debug(f"Email condition check: Enabled={config.ENABLE_EMAIL_NOTIFICATIONS}, Saved>0={len(batch_stats['saved_ideas']) > 0}, Errors>0={batch_stats['errors'] > 0}. Result: {send_email_condition}")
            if send_email_condition:
                logging.info(f"Condition met (Saved: {len(batch_stats['saved_ideas'])}, Errors: {batch_stats['errors']}), attempting to send summary email...")
                email_subject = f"SaaS Automator Run {run_count} Summary"
                email_body = f"Run {run_count} completed in {batch_duration:.2f}s.\nProcessed: {batch_stats['processed']}. Errors: {batch_stats['errors']}.\n"
                if batch_stats['saved_ideas']: email_body += f"Saved {len(batch_stats['saved_ideas'])} ideas (Threshold: {config.RATING_THRESHOLD}):\n" + "\n".join([f"- {s}" for s in batch_stats['saved_ideas']])
                else: email_body += "No new ideas met threshold."
                notifications.send_summary_email(email_subject, email_body)
            elif config.ENABLE_EMAIL_NOTIFICATIONS: logging.info("Skipping summary email: No saved ideas or errors this run.")
            status_message = f"Finished Run {run_count}. Processed {batch_stats['processed']}. Saved: {len(batch_stats['saved_ideas'])}. Errors: {batch_stats['errors']}."
            await api_clients.ping_uptime_kuma(session, message=status_message, ping_value=int(batch_duration))


            if run_count < config.MAX_RUNS: logging.info(f"--- Waiting {config.WAIT_BETWEEN_BATCHES}s ---"); await asyncio.sleep(config.WAIT_BETWEEN_BATCHES)

    logging.info("SaaS Idea Automator finished.")

if __name__ == "__main__":
    config.setup_logging()
    main_success = False
    try:
        if config.ENABLE_EMAIL_NOTIFICATIONS:
             if not all([config.SMTP_SERVER, config.SMTP_PORT, config.SMTP_USER, config.SMTP_PASSWORD, config.EMAIL_SENDER, config.EMAIL_RECIPIENT]):
                  logging.warning("Email notifications enabled but SMTP settings missing.")
        try:
            # Ensure numpy is imported if not already done elsewhere implicitly
            import numpy
            import aiohttp, dotenv, requests, sqlite3, nltk, sentence_transformers, sklearn
            logging.debug("All required libraries seem installed.")
        except ImportError as import_err:
             logging.error(f"Missing required library: {import_err.name}")
             logging.error(f"Install dependencies: pip install -r requirements.txt")
             if import_err.name == "nltk": logging.error("Also run: python -m nltk.downloader punkt stopwords")
             sys.exit(1)
        asyncio.run(main())
        main_success = True
    except KeyboardInterrupt: logging.warning("Script interrupted by user.")
    except Exception as e:
        logging.critical(f"Unhandled critical exception: {e}", exc_info=True)
        if not main_success and config.ENABLE_EMAIL_NOTIFICATIONS and all([config.SMTP_SERVER, config.SMTP_PORT, config.SMTP_USER, config.SMTP_PASSWORD, config.EMAIL_SENDER, config.EMAIL_RECIPIENT]):
             try:
                  error_traceback = traceback.format_exc()
                  fail_subject = "SaaS Automator CRITICAL FAILURE"
                  fail_body = f"Script stopped due to unhandled exception.\n\nError:\n{e}\n\nTraceback:\n{error_traceback}"
                  notifications.send_summary_email(fail_subject, fail_body)
             except Exception as email_fail_exc: logging.error(f"Failed to send critical failure email: {email_fail_exc}")
        sys.exit(1)
