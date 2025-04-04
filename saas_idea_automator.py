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

# Import project modules
import config
import api_clients
import state_manager
import analysis_utils
import notifications
import generation_strategies # New module
import evaluation_pipeline # New module

# Import specific functions if needed for clarity or direct use in main
# (Example: if helper functions were kept in main)
# from generation_strategies import get_prompt_type_identifier, softmax # Already moved

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
    current_good_examples = [] # Cache loaded examples

    async with aiohttp.ClientSession() as session:
        while run_count < config.MAX_RUNS:
            run_count += 1; batch_start_time = time.time()
            logging.info(f"===== Starting Run {run_count}/{config.MAX_RUNS} =====")
            batch_stats = {'processed': 0, 'saved_ideas': [], 'errors': 0}
            current_batch_prompt_types = {} # Track prompt type for ideas in this batch

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
                     high_rated_ideas_data = state_manager.get_high_rated_ideas(threshold=config.RATING_THRESHOLD, limit=100)
                     if high_rated_ideas_data and len(high_rated_ideas_data) >= config.TREND_ANALYSIS_MIN_IDEAS:
                          promising_themes = analysis_utils.get_combined_themes(high_rated_ideas_data)
                     else: logging.info("Skipping trend analysis: Not enough high-rated ideas meeting threshold."); promising_themes = []
            else: logging.info(f"Initial explore ratio: {current_explore_ratio:.2f}"); promising_themes = []

            # --- Automated Example Generation (Periodic) ---
            if config.ENABLE_AUTO_EXAMPLE_GENERATION and (run_count == 1 or (run_count -1) % config.AUTO_EXAMPLE_RUN_INTERVAL == 0):
                logging.info("Attempting to generate new examples for prompts...")
                # Use threshold=7.5 specifically for sourcing examples, even if main threshold changes later
                top_ideas = state_manager.get_high_rated_ideas(threshold=7.5, limit=config.AUTO_EXAMPLE_SOURCE_COUNT)
                if top_ideas and len(top_ideas) > 1:
                    idea_list_str = "\n".join([f"- {idea['name']}: {idea.get('description', 'N/A')}" for idea in top_ideas])
                    example_prompt = config.EXAMPLE_SYNTHESIS_PROMPT_TEMPLATE.format(
                        num_examples=config.AUTO_EXAMPLE_TARGET_COUNT, idea_list_str=idea_list_str)
                    example_response = await api_clients.generate_new_examples_async(session, example_prompt)
                    if example_response:
                        new_examples = [match.group(1).strip() for line in example_response.strip().split('\n') if (match := re.match(r"^\d+\.\s*(.*)", line.strip()))]
                        if new_examples:
                            logging.info(f"Synthesized new examples: {new_examples}")
                            state_manager.save_generated_examples(new_examples)
                            current_good_examples = new_examples # Update cache
                        else: logging.warning("Failed to parse synthesized examples.")
                    else: logging.warning("Example synthesis API call failed.")
                else: logging.info("Not enough high-rated ideas found to synthesize new examples yet.")
            # Load examples (either newly generated or from file)
            if not current_good_examples and config.ENABLE_AUTO_EXAMPLE_GENERATION: # Load only if enabled and not just generated
                 current_good_examples = state_manager.load_generated_examples()

            # Format examples for prompt injection
            good_examples_str = ""
            if current_good_examples:
                 good_examples_str = "\n\nGood examples based on past successes:\n- " + "\n- ".join(current_good_examples)


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
                # Note: Regenerated ideas don't have a direct 'source_prompt_type'
            else: # No pending ideas, generate new ones
                if config.ENABLE_MULTI_STEP_GENERATION:
                    initial_ideas, current_batch_prompt_types, generation_mode = await generation_strategies.generate_multi_step_ideas(session, avoid_keywords_str, avoid_examples_str)

                if not initial_ideas: # Fallback or Variation/New generation
                    if config.ENABLE_VARIATION_GENERATION and random.random() < config.VARIATION_GENERATION_PROBABILITY:
                        initial_ideas, current_batch_prompt_types, generation_mode = await generation_strategies.generate_variation_ideas(session)

                    if not initial_ideas: # Fallback to standard new idea generation
                        initial_ideas, current_batch_prompt_types, generation_mode = await generation_strategies.generate_new_ideas(
                            session, avoid_keywords_str, avoid_examples_str, good_examples_str, promising_themes, current_explore_ratio)


            # --- Generate Descriptions ---
            if not initial_ideas: logging.warning("No initial ideas generated."); await asyncio.sleep(10); continue
            idea_descriptions = await evaluation_pipeline.generate_descriptions_for_batch(session, initial_ideas)

            # --- Semantic Negative Feedback Filter ---
            ideas_after_neg_filter, candidate_embeddings_dict = await evaluation_pipeline.filter_semantic_duplicates(initial_ideas, idea_descriptions)

            # --- Self-Critique Step ---
            ideas = await evaluation_pipeline.run_self_critique(session, ideas_after_neg_filter)

            # --- Process Final Ideas ---
            if not ideas: logging.warning("No ideas to process."); await asyncio.sleep(10); continue
            logging.info(f"Attempting to process {len(ideas)} final ideas.")
            final_idea_embeddings_map = {}; ideas_to_embed_final = []; final_descriptions_map = {}
            for idea in ideas:
                 final_descriptions_map[idea] = idea_descriptions.get(idea)
                 # Use embeddings generated during semantic filtering if available
                 if idea in candidate_embeddings_dict and candidate_embeddings_dict[idea] is not None:
                      final_idea_embeddings_map[idea] = candidate_embeddings_dict[idea]
                 # Otherwise, queue for embedding generation if description exists
                 elif final_descriptions_map[idea]:
                      ideas_to_embed_final.append(final_descriptions_map[idea])
                 else: final_idea_embeddings_map[idea] = None # No description, no embedding

            if ideas_to_embed_final:
                 logging.info(f"Generating embeddings for {len(ideas_to_embed_final)} final idea descriptions...")
                 new_embeddings = analysis_utils.generate_embeddings(ideas_to_embed_final)
                 names_needing_embedding = [idea for idea in ideas if idea not in final_idea_embeddings_map]
                 if new_embeddings and len(new_embeddings) == len(names_needing_embedding):
                      for name, embedding in zip(names_needing_embedding, new_embeddings): final_idea_embeddings_map[name] = embedding
                 else:
                      logging.error("Failed to generate embeddings for some final ideas.");
                      for name in names_needing_embedding: final_idea_embeddings_map[name] = None


            # Create tasks for the evaluation pipeline's processing function
            tasks = [asyncio.create_task(evaluation_pipeline.process_single_idea_evaluation(
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
