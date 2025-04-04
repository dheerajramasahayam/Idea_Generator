import logging
import random
import re
import numpy as np
import asyncio

import config
import api_clients
import state_manager
import analysis_utils # For tokenize_and_clean if needed by variation/regen later

# --- Helper Functions (Copied from saas_idea_automator.py) ---

def get_prompt_type_identifier(template_content):
    """Gets the identifier ('general', 'platform', etc.) for a given prompt template string."""
    return config.PROMPT_TYPE_MAP.get(template_content, "unknown")

def softmax(x, temperature=1.0):
    """Compute softmax values for each sets of scores in x."""
    if not isinstance(x, np.ndarray): x = np.array(x)
    if temperature <= 0: # Handle zero or negative temperature -> greedy selection
        probs = np.zeros_like(x)
        if x.size > 0: probs[np.argmax(x)] = 1.0
        return probs
    x_shifted = x - np.max(x) # Improve numerical stability
    e_x = np.exp(x_shifted / temperature)
    sum_ex = e_x.sum(axis=0)
    if sum_ex == 0: return np.ones_like(x) / x.size # Return uniform distribution if sum is zero
    return e_x / sum_ex

# --- Generation Strategy Functions ---

async def generate_multi_step_ideas(session, avoid_keywords_str, avoid_examples_str):
    """Generates ideas using the Multi-Step (Concept -> Select -> Specific) strategy."""
    initial_ideas = []
    batch_prompt_types = {} # Track source type for performance analysis
    logging.info("Attempting multi-step generation strategy...")
    generation_mode = "Multi-Step"

    concepts_prompt = config.CONCEPT_GENERATION_PROMPT_TEMPLATE.format(num_concepts=config.NUM_CONCEPTS_TO_GENERATE)
    concepts_text = await api_clients.call_gemini_api_async(session, concepts_prompt)
    if not concepts_text:
        logging.warning("Concept generation call failed.")
        return initial_ideas, batch_prompt_types, "Fallback" # Return empty and fallback mode

    concepts = [match.group(1).strip() for line in concepts_text.strip().split('\n') if (match := re.match(r"^\d+\.\s*(.*)", line.strip()))]
    logging.info(f"Generated {len(concepts)} concepts.")
    if not concepts:
        logging.warning("Failed to parse generated concepts.")
        return initial_ideas, batch_prompt_types, "Fallback"

    selection_prompt = config.CONCEPT_SELECTION_PROMPT_TEMPLATE.format(num_to_select=config.NUM_CONCEPTS_TO_SELECT, concept_list_str="\n".join([f"{i+1}. {c}" for i, c in enumerate(concepts)]))
    selected_concepts_text = await api_clients.call_gemini_api_async(session, selection_prompt)
    if not selected_concepts_text:
        logging.warning("Concept selection call failed.")
        return initial_ideas, batch_prompt_types, "Fallback"

    selected_concept_names = [match.group(1).strip() for line in selected_concepts_text.strip().split('\n') if (match := re.match(r"^\d+\.\s*(.*)", line.strip()))]
    valid_selected_concepts = [c for c in selected_concept_names if c in concepts]
    logging.info(f"Selected {len(valid_selected_concepts)} concepts: {valid_selected_concepts}")
    if not valid_selected_concepts:
        logging.warning("Concept selection failed to return valid concepts.")
        return initial_ideas, batch_prompt_types, "Fallback"

    specific_idea_tasks = []
    for concept in valid_selected_concepts:
        # NOTE: SPECIFIC_IDEA_GENERATION prompt doesn't have {good_examples_section} placeholder
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
            for idea in parsed_ideas: batch_prompt_types[idea] = "multi-step" # Assign type

    logging.info(f"Generated {len(initial_ideas)} specific ideas via multi-step.")
    if not initial_ideas: generation_mode = "Fallback" # Fallback if generation failed

    return initial_ideas, batch_prompt_types, generation_mode


async def generate_variation_ideas(session):
    """Generates ideas using the Variation strategy."""
    initial_ideas = []
    batch_prompt_types = {}
    generation_mode = "Variations"
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
            num_variations=config.NUM_VARIATIONS_TO_GENERATE)
        variation_response = await api_clients.generate_variation_ideas_async(session, variation_prompt)
        if variation_response:
            parsed_variations = [match.group(1).strip() for line in variation_response.strip().split('\n') if (match := re.match(r"^\d+\.\s*(.*)", line.strip()))]
            initial_ideas = [i for i in parsed_variations if 0 < len(i) < 200]
            logging.info(f"Generated {len(initial_ideas)} variations.")
            for idea in initial_ideas: batch_prompt_types[idea] = "variation" # Assign type
        else:
            logging.warning("Variation generation failed. Falling back.")
            generation_mode = "Fallback"
    else:
        logging.info("No suitable candidates for variation. Generating new ideas.")
        generation_mode = "Fallback" # Fallback to new ideas

    return initial_ideas, batch_prompt_types, generation_mode


async def generate_new_ideas(session, avoid_keywords_str, avoid_examples_str, good_examples_str, promising_themes, current_explore_ratio):
    """Generates new ideas using dynamic or random prompt selection."""
    initial_ideas = []
    batch_prompt_types = {}
    generation_mode = "New Ideas"
    source_prompt_type_for_batch = "unknown" # Default

    explore = random.random() < current_explore_ratio
    base_prompt_template = None

    # --- Dynamic Prompt Selection Logic ---
    if config.ENABLE_DYNAMIC_PROMPT_SELECTION:
        logging.info("Attempting dynamic prompt selection...")
        perf_data = state_manager.get_prompt_performance()
        available_templates = config.IDEA_GENERATION_PROMPT_TEMPLATES
        template_types = [get_prompt_type_identifier(t) for t in available_templates]
        scores = []
        default_score = 5.0 # Assign a neutral score for exploration
        for p_type in template_types:
            data = perf_data.get(p_type)
            if data and data['count'] >= config.DYNAMIC_SELECTION_MIN_DATA:
                scores.append(data['avg_rating'])
            else: scores.append(default_score)
        if not scores or not available_templates:
             logging.error("No available templates or scores for dynamic selection. Falling back to random.")
             base_prompt_template = random.choice(config.IDEA_GENERATION_PROMPT_TEMPLATES) if config.IDEA_GENERATION_PROMPT_TEMPLATES else None
        else:
             probabilities = softmax(scores, temperature=config.DYNAMIC_SELECTION_TEMP)
             logging.debug(f"Prompt types: {template_types}, Scores: {scores}, Probs: {probabilities}")
             base_prompt_template = random.choices(available_templates, weights=probabilities, k=1)[0]
        source_prompt_type_for_batch = get_prompt_type_identifier(base_prompt_template) if base_prompt_template else "unknown"
        logging.info(f"Dynamically selected prompt type: '{source_prompt_type_for_batch}'")
    else: # Standard random selection
        base_prompt_template = random.choice(config.IDEA_GENERATION_PROMPT_TEMPLATES) if config.IDEA_GENERATION_PROMPT_TEMPLATES else None
        source_prompt_type_for_batch = get_prompt_type_identifier(base_prompt_template) if base_prompt_template else "unknown"
    # --- End Dynamic Selection ---

    if base_prompt_template:
        logging.info(f"Using generation prompt type: {source_prompt_type_for_batch}")
        positive_themes_str = ""
        if not explore and promising_themes:
            logging.info(f"Exploiting promising themes: {promising_themes}")
            positive_themes_str = "\n\nTry to generate ideas related to these concepts: " + ", ".join([f"'{t}'" for t in promising_themes])
        elif explore: logging.info("Exploring with broader prompt.")
        else: logging.info("No promising themes yet or exploring.")

        # Inject dynamic examples
        generation_prompt = base_prompt_template.format(
            num_ideas=config.IDEAS_PER_BATCH,
            avoid_keywords_section=avoid_keywords_str,
            avoid_examples_section=avoid_examples_str,
            good_examples_section=good_examples_str # Add generated examples
        )
        generation_prompt += positive_themes_str
        initial_ideas = await generate_ideas(session, generation_prompt)
        for idea in initial_ideas: batch_prompt_types[idea] = source_prompt_type_for_batch # Assign type
    else:
         logging.error("Could not select a base prompt template.")

    return initial_ideas, batch_prompt_types, generation_mode

# Helper function moved from main script
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
