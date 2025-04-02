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
import config
import api_clients
import state_manager
import analysis_utils
import notifications
import nltk
import sentence_transformers
import numpy
import sklearn

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

async def research_idea(session, idea_name):
    """Performs web searches using the configured API and compiles a summary."""
    logging.info(f">>> Researching idea: '{idea_name}'...")
    queries = await get_search_queries(session, idea_name)
    research_summary = ""
    search_tasks = [api_clients.call_search_api_async(session, query) for query in queries]
    search_results_list = await asyncio.gather(*search_tasks, return_exceptions=True)
    for query, result_or_exc in zip(queries, search_results_list):
        if isinstance(result_or_exc, Exception): logging.error(f"Search task failed for '{query}': {result_or_exc}"); continue
        search_results = result_or_exc
        if not search_results: continue
        research_summary += f"\nSearch Results for '{query}':\n"; count = 0; results_list = []
        provider = config.SEARCH_PROVIDER
        try:
            if provider == "google" and 'items' in search_results: results_list, keys = search_results['items'], ('title', 'snippet', 'link')
            elif provider == "serper" and 'organic' in search_results: results_list, keys = search_results['organic'], ('title', 'snippet', 'link')
            elif provider == "brave" and 'web' in search_results and 'results' in search_results['web']: results_list, keys = search_results['web']['results'], ('title', 'description', 'url')
            else: logging.warning(f"Unexpected search structure from {provider} for '{query}'."); continue
            for result in results_list:
                if count >= config.SEARCH_RESULTS_LIMIT: break
                title, snippet, link = result.get(keys[0], 'N/A'), result.get(keys[1], 'N/A'), result.get(keys[2], '#')
                entry = f"- {title}: {snippet} ({link})\n"
                if len(research_summary) + len(entry) < config.MAX_SUMMARY_LENGTH: research_summary += entry; count += 1
                else: logging.warning("Research summary truncated."); break
            research_summary += "-" * 10 + "\n"
        except Exception as parse_exc: logging.error(f"Error parsing search results for '{query}' from {provider}: {parse_exc}", exc_info=True)
    logging.info(f"Research complete for '{idea_name}'. Summary length: {len(research_summary)}")
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
    if not rating_context: return None, None
    prompt = config.RATING_PROMPT_TEMPLATE.format(idea_name=idea_name, rating_context=rating_context)
    response_text = await api_clients.call_gemini_api_async(session, prompt)
    if not response_text: return None, None
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
    if parsed_count != len(expected_keys): logging.error(f"Parsed {parsed_count}/5 scores. Missing: {set(expected_keys) - set(scores.keys())}. Resp: '{response_text}'"); return None, None
    weighted_total = sum(score * config.RATING_WEIGHTS.get(key, 1.0 / len(expected_keys)) for key, score in scores.items())
    logging.info(f"Received rating breakdown for '{idea_name}': {scores}. Weighted Score: {weighted_total:.1f}")
    return weighted_total, justifications

async def process_single_idea(idea_name, idea_description, idea_embedding, processed_ideas_set, session, semaphore, batch_stats):
    """Async function to research, extract facts, rate, and save state for a single idea."""
    async with semaphore:
        idea_lower = idea_name.lower()
        if idea_lower in processed_ideas_set: return
        status, rating, justifications = "processing", None, {}
        try:
            status = "researching"; summary = await research_idea(session, idea_name)
            await asyncio.sleep(random.uniform(0.5, 1.0))
            status = "fact_extraction"; extracted_facts = await extract_facts_for_rating(session, idea_name, summary)
            rating_context = extracted_facts if extracted_facts else summary
            if not rating_context: raise ValueError("Empty rating context.")
            await asyncio.sleep(random.uniform(0.5, 1.0))
            status = "rating"; rating, justifications = await rate_idea(session, idea_name, rating_context)
            if rating is not None:
                status = "rated"
                if rating >= config.RATING_THRESHOLD:
                    state_manager.save_rated_idea(idea_name, rating, justifications if justifications else {})
                    batch_stats['saved_ideas'].append(f"{idea_name} ({rating:.1f})"); status = "saved"
            else: status = "rating_failed"
            logging.info(f"--- Finished processing idea: '{idea_name}' ---")
        except Exception as e: logging.error(f"Unexpected error processing idea '{idea_name}': {e}", exc_info=True); status = "error"; batch_stats['errors'] += 1
        finally:
            state_manager.update_idea_state(idea_name, status, rating, justifications, embedding=idea_embedding, description=idea_description)
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

            # --- Adaptive Explore/Exploit & Trend Analysis ---
            if run_count > 1:
                recent_ratings = state_manager.get_recent_ratings(limit=recent_ratings_window)
                if len(recent_ratings) >= 10:
                    success_rate = sum(1 for r in recent_ratings if r >= config.RATING_THRESHOLD) / len(recent_ratings)
                    logging.info(f"Recent success rate ({len(recent_ratings)} ideas): {success_rate:.2%}")
                    if success_rate > target_success_rate_upper: current_explore_ratio = max(min_explore, current_explore_ratio - adjustment_step); logging.info(f"Decreasing explore ratio to {current_explore_ratio:.2f}")
                    elif success_rate < target_success_rate_lower: current_explore_ratio = min(max_explore, current_explore_ratio + adjustment_step); logging.info(f"Increasing explore ratio to {current_explore_ratio:.2f}")
                    else: logging.info(f"Keeping explore ratio at {current_explore_ratio:.2f}")
                else: logging.info(f"Not enough recent ratings ({len(recent_ratings)}) to adjust explore ratio.")

                if (run_count - 1) % config.TREND_ANALYSIS_RUN_INTERVAL == 0:
                     high_rated_ideas_data = state_manager.get_high_rated_ideas(limit=100)
                     if high_rated_ideas_data and len(high_rated_ideas_data) >= config.TREND_ANALYSIS_MIN_IDEAS:
                          promising_themes = analysis_utils.get_combined_themes(high_rated_ideas_data)
                     else: logging.info("Skipping trend analysis: Not enough high-rated ideas."); promising_themes = []
            else: logging.info(f"Initial explore ratio: {current_explore_ratio:.2f}"); promising_themes = []

            # --- Idea Generation ---
            initial_ideas = []
            generation_mode = "New Ideas (Default)"

            # Prepare keyword avoidance string (always do this)
            avoid_keywords_str = ""
            low_rated_texts = state_manager.get_low_rated_texts(limit=50)
            if low_rated_texts:
                 avoid_keywords = analysis_utils.extract_keywords(low_rated_texts, top_n=15)
                 if avoid_keywords:
                      avoid_keywords_str = "\n\nAvoid ideas related to these keywords: " + ", ".join(avoid_keywords)

            # Decide generation strategy
            if config.ENABLE_MULTI_STEP_GENERATION:
                 logging.info("Attempting multi-step generation strategy...")
                 generation_mode = "Multi-Step"
                 concepts_prompt = config.CONCEPT_GENERATION_PROMPT_TEMPLATE.format(num_concepts=config.NUM_CONCEPTS_TO_GENERATE)
                 concepts_text = await api_clients.call_gemini_api_async(session, concepts_prompt)
                 if concepts_text:
                      concepts = [match.group(1).strip() for line in concepts_text.strip().split('\n') if (match := re.match(r"^\d+\.\s*(.*)", line.strip()))]
                      logging.info(f"Generated {len(concepts)} concepts.")
                      if concepts:
                           selection_prompt = config.CONCEPT_SELECTION_PROMPT_TEMPLATE.format(num_to_select=config.NUM_CONCEPTS_TO_SELECT, concept_list_str="\n".join([f"{i+1}. {c}" for i, c in enumerate(concepts)]))
                           selected_concepts_text = await api_clients.call_gemini_api_async(session, selection_prompt)
                           if selected_concepts_text:
                                selected_concept_names = [match.group(1).strip() for line in selected_concepts_text.strip().split('\n') if (match := re.match(r"^\d+\.\s*(.*)", line.strip()))]
                                valid_selected_concepts = [c for c in selected_concept_names if c in concepts]
                                logging.info(f"Selected {len(valid_selected_concepts)} concepts: {valid_selected_concepts}")
                                if valid_selected_concepts:
                                     specific_idea_tasks = []
                                     for concept in valid_selected_concepts:
                                          # Add keyword avoidance to specific idea prompt
                                          specific_prompt = config.SPECIFIC_IDEA_GENERATION_PROMPT_TEMPLATE.format(selected_concept=concept, num_ideas=config.NUM_IDEAS_PER_CONCEPT)
                                          specific_prompt += avoid_keywords_str # Append avoidance here too
                                          specific_idea_tasks.append(api_clients.call_gemini_api_async(session, specific_prompt))
                                     specific_idea_results = await asyncio.gather(*specific_idea_tasks, return_exceptions=True)
                                     for result_or_exc in specific_idea_results:
                                          if isinstance(result_or_exc, Exception): logging.error(f"Specific idea generation failed: {result_or_exc}")
                                          elif result_or_exc:
                                               parsed_ideas = [match.group(1).strip() for line in result_or_exc.strip().split('\n') if (match := re.match(r"^\d+\.\s*(.*)", line.strip())) and 0 < len(match.group(1).strip()) < 200]
                                               initial_ideas.extend(parsed_ideas)
                                     logging.info(f"Generated {len(initial_ideas)} specific ideas via multi-step.")
                                else: logging.warning("Concept selection failed to return valid concepts.")
                           else: logging.warning("Concept selection call failed.")
                      else: logging.warning("Failed to parse generated concepts.")
                 else: logging.warning("Concept generation call failed.")
                 if not initial_ideas: logging.warning("Multi-step generation failed. Falling back."); generation_mode = "Fallback"

            # Fallback or Variation/New generation if multi-step is disabled or failed
            if not initial_ideas:
                if config.ENABLE_VARIATION_GENERATION and random.random() < config.VARIATION_GENERATION_PROBABILITY:
                     logging.info("Attempting variation generation strategy...")
                     candidate_ideas = state_manager.get_variation_candidate_ideas(config.VARIATION_SOURCE_MIN_RATING, config.VARIATION_SOURCE_MAX_RATING, limit=5)
                     if candidate_ideas:
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
                              num_variations=config.NUM_VARIATIONS_TO_GENERATE
                          )
                          variation_response = await api_clients.generate_variation_ideas_async(session, variation_prompt)
                          if variation_response:
                               parsed_variations = [match.group(1).strip() for line in variation_response.strip().split('\n') if (match := re.match(r"^\d+\.\s*(.*)", line.strip()))]
                               initial_ideas = [i for i in parsed_variations if 0 < len(i) < 200]
                               logging.info(f"Generated {len(initial_ideas)} variations."); generation_mode = "Variations"
                          else: logging.warning("Variation generation failed. Falling back.")
                     else: logging.info("No suitable candidates for variation. Generating new ideas.")

                if not initial_ideas: # Fallback to standard new idea generation
                    generation_mode = "New Ideas"
                    explore = random.random() < current_explore_ratio
                    base_prompt_template = random.choice(config.IDEA_GENERATION_PROMPT_TEMPLATES)
                    logging.info(f"Using generation prompt type: {base_prompt_template[:50]}...")
                    # Prepare positive theme string (improved representation)
                    positive_themes_str = ""
                    if not explore and promising_themes:
                         logging.info(f"Exploiting promising themes: {promising_themes}")
                         positive_themes_str = "\n\nTry to generate ideas related to these concepts: " + ", ".join([f"'{t}'" for t in promising_themes])
                    elif explore: logging.info("Exploring with broader prompt.")
                    else: logging.info("No promising themes yet or exploring.")

                    # Format the chosen base prompt with avoidance and positive themes
                    generation_prompt = base_prompt_template.format(num_ideas=config.IDEAS_PER_BATCH, avoid_keywords_section=avoid_keywords_str)
                    generation_prompt += positive_themes_str

                    initial_ideas = await generate_ideas(session, generation_prompt)

            # --- Generate Descriptions ---
            if not initial_ideas: logging.warning("No initial ideas generated."); await asyncio.sleep(10); continue
            idea_descriptions = await generate_descriptions(session, initial_ideas)

            # --- Semantic Negative Feedback Filter ---
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
                        embedding = temp_embeddings_dict.get(idea_name)
                        description = idea_descriptions.get(idea_name)
                        if embedding and description:
                            if not analysis_utils.check_similarity(embedding, low_rated_embeddings, config.NEGATIVE_FEEDBACK_SIMILARITY_THRESHOLD):
                                ideas_after_neg_filter.append(idea_name); candidate_embeddings_dict[idea_name] = embedding
                            else: logging.info(f"Filtered out '{idea_name}' due to description similarity.")
                        elif description: ideas_after_neg_filter.append(idea_name); candidate_embeddings_dict[idea_name] = None
                else:
                    logging.warning("Could not generate embeddings. Skipping semantic filter.")
                    ideas_after_neg_filter = initial_ideas; candidate_embeddings_dict = {name: None for name in initial_ideas}
            else:
                 logging.info("No low-rated embeddings or no valid descriptions. Skipping semantic filter.")
                 ideas_after_neg_filter = initial_ideas; candidate_embeddings_dict = {name: None for name in initial_ideas}

            # --- Self-Critique Step ---
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

            tasks = [asyncio.create_task(process_single_idea(idea, final_descriptions_map.get(idea), final_idea_embeddings_map.get(idea), processed_ideas_set, session, semaphore, batch_stats)) for idea in ideas if idea.lower() not in processed_ideas_set]
            if not tasks: logging.warning("No new ideas to process after filtering processed.");
            else: logging.info(f"Created {len(tasks)} tasks for new ideas."); await asyncio.gather(*tasks)

            # --- Finish Run & Notify ---
            batch_duration = time.time() - batch_start_time
            logging.info(f"===== Finished Run {run_count}/{config.MAX_RUNS} in {batch_duration:.2f} seconds =====")
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
            import aiohttp, dotenv, requests, sqlite3, nltk, sentence_transformers, numpy, sklearn
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
