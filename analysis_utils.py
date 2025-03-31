import re
import logging
from collections import Counter
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords

# --- NLTK Data Download ---
# Ensure necessary NLTK data is downloaded once
try:
    nltk.data.find('tokenizers/punkt')
    logging.debug("NLTK 'punkt' resource found.")
except LookupError: # Correct exception type
    logging.info("NLTK 'punkt' resource not found. Downloading...")
    try:
        nltk.download('punkt', quiet=True)
        logging.info("Successfully downloaded NLTK 'punkt'.")
    except Exception as download_exc:
        logging.error(f"Failed to download NLTK 'punkt': {download_exc}. Please install manually.")
        # Optionally raise error or exit if essential
try:
    nltk.data.find('corpora/stopwords')
    logging.debug("NLTK 'stopwords' resource found.")
except LookupError: # Correct exception type
    logging.info("NLTK 'stopwords' resource not found. Downloading...")
    try:
        nltk.download('stopwords', quiet=True)
        logging.info("Successfully downloaded NLTK 'stopwords'.")
    except Exception as download_exc:
        logging.error(f"Failed to download NLTK 'stopwords': {download_exc}. Please install manually.")
        # Optionally raise error or exit if essential

# --- Constants ---
# Use NLTK's stopwords and add custom ones
NLTK_STOPWORDS = set(stopwords.words('english'))
CUSTOM_STOPWORDS = set([
    # Add common words from previous list + potential noise
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
ALL_STOP_WORDS = NLTK_STOPWORDS.union(CUSTOM_STOPWORDS)

# --- Functions ---

def tokenize_and_clean(text):
    """Tokenizes text, converts to lowercase, and removes stop words and short words."""
    if not text:
        return []
    try:
        words = nltk.word_tokenize(text.lower())
        # Keep words that are alphabetic, not stop words, and longer than 2 chars
        return [word for word in words if word.isalpha() and word not in ALL_STOP_WORDS and len(word) > 2]
    except Exception as e:
        logging.error(f"Error tokenizing text: {e}")
        return []

def generate_ngrams(tokens, n):
    """Generates n-grams from a list of tokens."""
    return [" ".join(gram) for gram in ngrams(tokens, n)]

def get_promising_themes(high_rated_ideas_data, top_n=7):
    """
    Analyzes high-rated ideas from DB data and returns top keywords/themes (including n-grams).
    Args:
        high_rated_ideas_data (list): List of dicts, each like {'name': 'Idea Name', 'rating': 8.0}
        top_n (int): Number of top themes to return.
    Returns:
        list: List of top N keywords/phrases.
    """
    if not high_rated_ideas_data:
        return []

    logging.info(f"Analyzing {len(high_rated_ideas_data)} high-rated ideas for trends using N-grams...")
    all_features = []
    for item in high_rated_ideas_data:
        idea_name = item.get('name')
        if idea_name:
            tokens = tokenize_and_clean(idea_name)
            if tokens:
                all_features.extend(tokens) # Add single keywords
                all_features.extend(generate_ngrams(tokens, 2)) # Add bigrams
                all_features.extend(generate_ngrams(tokens, 3)) # Add trigrams

    if not all_features:
        logging.warning("No meaningful keywords or n-grams extracted from high-rated idea names.")
        return []

    feature_counts = Counter(all_features)

    # Filter out features that only appear once (often noise)
    filtered_counts = {feature: count for feature, count in feature_counts.items() if count > 1}
    if not filtered_counts:
        logging.warning("No features appeared more than once. Returning most common single features.")
        # Fallback to most common single features if no n-grams repeat
        filtered_counts = feature_counts

    # Sort by frequency
    sorted_features = sorted(filtered_counts.items(), key=lambda item: item[1], reverse=True)

    promising_themes = [feature for feature, count in sorted_features[:top_n]]
    logging.info(f"Identified promising keywords/themes (N-grams): {promising_themes}")
    return promising_themes
