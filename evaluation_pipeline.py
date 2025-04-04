import asyncio
import logging
import re
import random

import config
import api_clients
import state_manager
import analysis_utils # For embedding/similarity

async def generate_descriptions_for_batch(session, idea_names):
    """Generates descriptions for a list of idea names using Gemini."""
    # (Code moved from saas_idea_automator.generate_descriptions)
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

async def filter_semantic_duplicates(idea_names, idea_descriptions):
    """Filters ideas based on semantic similarity to low-rated ideas."""
    # (Logic moved from saas_idea_automator.main - Semantic Negative Feedback Filter section)
    ideas_after_filter = []
    candidate_embeddings_dict = {} # Store embeddings for ideas that pass
    texts_to_embed = [idea_descriptions.get(name) for name in idea_names if idea_descriptions.get(name)]
    names_with_desc = [name for name in idea_names if idea_descriptions.get(name)]
    low_rated_embeddings = state_manager.get_low_rated_embeddings(limit=200)

    if low_rated_embeddings and texts_to_embed:
        logging.info("Applying semantic negative feedback filter based on descriptions...")
        candidate_embeddings_list = analysis_utils.generate_embeddings(texts_to_embed)
        if candidate_embeddings_list and len(candidate_embeddings_list) == len(texts_to_embed):
            temp_embeddings_dict = {name: emb for name, emb in zip(names_with_desc, candidate_embeddings_list)}
            for idea_name in idea_names:
                embedding = temp_embeddings_dict.get(idea_name)
                description = idea_descriptions.get(idea_name)
                # Keep idea if it has no description/embedding OR if it's not too similar
                if not (embedding and description) or not analysis_utils.check_similarity(embedding, low_rated_embeddings, config.NEGATIVE_FEEDBACK_SIMILARITY_THRESHOLD):
                    ideas_after_filter.append(idea_name)
                    # Store embedding if it exists, otherwise store None
                    candidate_embeddings_dict[idea_name] = embedding if (embedding and description) else None
                else:
                    logging.info(f"Filtered out '{idea_name}' due to description similarity.")
                    candidate_embeddings_dict[idea_name] = None # Ensure filtered ideas don't carry embedding forward
        else:
            logging.warning("Could not generate embeddings for semantic filter. Skipping filter.")
            ideas_after_filter = idea_names
            candidate_embeddings_dict = {name: None for name in idea_names} # No embeddings available
    else:
         logging.info("No low-rated embeddings or no valid descriptions for new ideas. Skipping semantic filter.")
         ideas_after_filter = idea_names
         candidate_embeddings_dict = {name: None for name in idea_names} # No embeddings available

    return ideas_after_filter, candidate_embeddings_dict


async def run_self_critique(session, idea_names):
    """Performs self-critique step on a list of idea names."""
    # (Logic moved from saas_idea_automator.main - Self-Critique Step section)
    if not idea_names: return []
    logging.info(f"Performing self-critique on {len(idea_names)} ideas...")
    ideas_passed_critique = []
    idea_list_str = "\n".join([f"{i+1}. {idea}" for i, idea in enumerate(idea_names)])
    critique_prompt = config.SELF_CRITIQUE_PROMPT_TEMPLATE.format(idea_list_str=idea_list_str)
    critique_response = await api_clients.call_gemini_api_async(session, critique_prompt)

    if critique_response and critique_response.lower().strip() != "none":
        critiqued_ideas_list = [match.group(1).strip() for line in critique_response.strip().split('\n') if (match := re.match(r"^\d+\.\s*(.*)", line.strip())) and 0 < len(match.group(1).strip()) < 200]
        original_ideas_lower = {i.lower() for i in idea_names}
        # Filter the critique list to only include ideas that were actually in the input list
        ideas_passed_critique = [idea for idea in critiqued_ideas_list if idea.lower() in original_ideas_lower]
        logging.info(f"Self-critique resulted in {len(ideas_passed_critique)} ideas passing filter.")
    elif critique_response and critique_response.lower().strip() == "none":
        logging.info("Self-critique resulted in no ideas passing filter.")
        ideas_passed_critique = []
    else:
        logging.warning("Self-critique failed/empty. Using pre-critique list.")
        ideas_passed_critique = idea_names # Pass all if critique fails

    return ideas_passed_critique


async def process_single_idea_evaluation(idea_name, idea_description, idea_embedding, source_prompt_type, processed_ideas_set, session, semaphore, batch_stats):
    """Async function to research, extract facts, rate, save state, update performance, and potentially queue re-generation."""
    # (Renamed from process_single_idea in saas_idea_automator.py)
    async with semaphore:
        idea_lower = idea_name.lower()
        if idea_lower in processed_ideas_set: return
        status, rating, justifications, scores = "processing", None, {}, None
        try:
            status = "researching"; summary = await research_idea(session, idea_name, idea_description) # research_idea remains separate for now
            await asyncio.sleep(random.uniform(0.5, 1.0))
            status = "fact_extraction"; extracted_facts = await extract_facts_for_rating(session, idea_name, summary) # extract_facts remains separate
            rating_context = extracted_facts if extracted_facts else summary
            if not rating_context: raise ValueError("Empty rating context.")
            await asyncio.sleep(random.uniform(0.5, 1.0))
            status = "rating"; rating, justifications, scores = await rate_idea(session, idea_name, rating_context) # rate_idea remains separate
            if rating is not None:
                status = "rated"
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

# --- Functions still needed by process_single_idea_evaluation (kept separate for now) ---
# These could potentially be moved here too, but research_idea is quite long.

async def research_idea(session, idea_name, idea_description):
    """Performs web searches and optionally GitHub search, compiling a summary."""
    # (Code is identical to the one in saas_idea_automator.py - keeping it there for now)
    # This function needs access to api_clients.call_search_api_async and call_github_search_api_async
    # It also needs _is_dev_tool_idea and get_search_queries
    # For modularity, these dependencies would need to be passed or imported.
    # Let's assume it's called from the main script for now.
    # To make this truly modular, research_idea would need refactoring.
    pass # Placeholder - actual call will be from main script

async def extract_facts_for_rating(session, idea_name, research_summary):
    """Uses Gemini to extract key facts relevant to rating criteria."""
    # (Code is identical to the one in saas_idea_automator.py - keeping it there for now)
    pass # Placeholder

async def rate_idea(session, idea_name, rating_context):
    """Rates an idea based on provided context (extracted facts or raw summary) using Gemini."""
    # (Code is identical to the one in saas_idea_automator.py - keeping it there for now)
    pass # Placeholder

# We also need _is_dev_tool_idea and get_search_queries if research_idea is moved here.

# --- Temporary Placeholder Functions (to avoid breaking process_single_idea_evaluation) ---
# These will be replaced by calls to the actual functions in the main script later.

async def research_idea(session, idea_name, idea_description):
     # This is complex, involves get_search_queries, web search, github search
     # Keep this logic in the main orchestrator for now.
     logging.warning("research_idea called within evaluation_pipeline (should be called from orchestrator)")
     return f"Dummy research summary for {idea_name}"

async def extract_facts_for_rating(session, idea_name, research_summary):
     logging.warning("extract_facts_for_rating called within evaluation_pipeline (should be called from orchestrator)")
     # Call the actual API client function
     return await api_clients.call_gemini_api_async(session, config.FACT_EXTRACTION_PROMPT_TEMPLATE.format(idea_name=idea_name, research_summary=research_summary))


async def rate_idea(session, idea_name, rating_context):
     logging.warning("rate_idea called within evaluation_pipeline (should be called from orchestrator)")
     # Call the actual API client function and parse
     response_text = await api_clients.call_gemini_api_async(session, config.RATING_PROMPT_TEMPLATE.format(idea_name=idea_name, rating_context=rating_context))
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
             except ValueError: pass # Ignore conversion errors
     if parsed_count != len(expected_keys): return None, None, None
     weighted_total = sum(score * config.RATING_WEIGHTS.get(key, 1.0 / len(expected_keys)) for key, score in scores.items())
     return weighted_total, justifications, scores
