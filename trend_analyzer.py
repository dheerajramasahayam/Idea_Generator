import re
import logging
from collections import Counter
import config # To get the OUTPUT_FILE path

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
    "for", "with", "and", "new", "get", "use", "best", "top", "simple", "advanced"
])

def parse_ideas_from_file(filename=config.OUTPUT_FILE):
    """Parses high-scoring idea names from the markdown output file."""
    ideas = []
    # Regex to find lines starting with '##' followed by the idea and score
    # Example: ## Idea Name - Score: 8.1
    idea_pattern = re.compile(r"^\s*##\s*(.+?)\s*-\s*Score:\s*[\d\.]+\s*$", re.IGNORECASE)

    if not os.path.exists(filename):
        logging.error(f"Output file '{filename}' not found. Cannot analyze trends.")
        return ideas

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                match = idea_pattern.match(line.strip())
                if match:
                    idea_name = match.group(1).strip()
                    if idea_name:
                        ideas.append(idea_name)
    except Exception as e:
        logging.error(f"Error reading or parsing output file '{filename}': {e}")

    return ideas

def extract_keywords(text):
    """Extracts simple keywords from text, removing stop words."""
    words = re.findall(r'\b\w+\b', text.lower())
    return [word for word in words if word not in STOP_WORDS and len(word) > 2] # Ignore very short words

def analyze_trends(ideas, top_n=15):
    """Performs basic keyword frequency analysis on idea names."""
    if not ideas:
        logging.warning("No ideas found in the output file to analyze.")
        return

    logging.info(f"Analyzing trends based on {len(ideas)} high-scoring ideas...")
    all_keywords = []
    for idea in ideas:
        all_keywords.extend(extract_keywords(idea))

    if not all_keywords:
        logging.warning("No meaningful keywords extracted from the idea names.")
        return

    keyword_counts = Counter(all_keywords)

    print("\n--- Trend Analysis: Top Keywords in High-Scoring Ideas ---")
    if not keyword_counts:
        print("No keywords found to analyze.")
    else:
        print(f"Found {len(keyword_counts)} unique keywords. Top {top_n}:")
        for keyword, count in keyword_counts.most_common(top_n):
            print(f"- {keyword}: {count}")
    print("----------------------------------------------------------")


if __name__ == "__main__":
    config.setup_logging() # Use logging setup from config
    logging.info("Starting Trend Analyzer...")
    high_scoring_ideas = parse_ideas_from_file()
    analyze_trends(high_scoring_ideas)
    logging.info("Trend Analyzer finished.")
