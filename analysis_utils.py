import re
import logging
from collections import Counter
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import json # For loading embeddings from DB
import config
# Import sklearn components if available, handle ImportError
try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    logging.warning("scikit-learn library not found. LDA and Clustering analysis will be disabled.")
    SKLEARN_AVAILABLE = False

# --- Global Variable for Embedding Model ---
embedding_model = None
try:
    device_to_use = 'cpu'
    logging.info(f"Loading sentence transformer model: {config.EMBEDDING_MODEL} onto device: {device_to_use}...")
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL, device=device_to_use)
    logging.info("Sentence transformer model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load sentence transformer model '{config.EMBEDDING_MODEL}': {e}", exc_info=True)

# --- NLTK Data Download ---
# (Keep the existing NLTK download logic)
try:
    nltk.data.find('tokenizers/punkt')
    logging.debug("NLTK 'punkt' resource found.")
except LookupError:
    logging.info("NLTK 'punkt' resource not found. Downloading...")
    try: nltk.download('punkt', quiet=True); logging.info("Successfully downloaded NLTK 'punkt'.")
    except Exception as download_exc: logging.error(f"Failed to download NLTK 'punkt': {download_exc}. Please install manually.")
try:
    nltk.data.find('corpora/stopwords')
    logging.debug("NLTK 'stopwords' resource found.")
except LookupError:
    logging.info("NLTK 'stopwords' resource not found. Downloading...")
    try: nltk.download('stopwords', quiet=True); logging.info("Successfully downloaded NLTK 'stopwords'.")
    except Exception as download_exc: logging.error(f"Failed to download NLTK 'stopwords': {download_exc}. Please install manually.")

# --- Constants ---
NLTK_STOPWORDS = set(stopwords.words('english'))
CUSTOM_STOPWORDS = set([
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

# --- Helper Functions ---

def tokenize_and_clean(text):
    """Tokenizes text, converts to lowercase, and removes stop words and short words."""
    if not text: return []
    try:
        words = nltk.word_tokenize(text.lower())
        return [word for word in words if word.isalpha() and word not in ALL_STOP_WORDS and len(word) > 2]
    except Exception as e:
        logging.error(f"Error tokenizing text: {e}"); return []

def generate_ngrams(tokens, n):
    """Generates n-grams from a list of tokens."""
    return [" ".join(gram) for gram in ngrams(tokens, n)]

def generate_embeddings(texts):
    """Generates embeddings for a list of texts using the loaded model."""
    if embedding_model is None: return None
    if not texts: return []
    try:
        logging.debug(f"Generating embeddings for {len(texts)} texts using device: {embedding_model.device}...")
        if not isinstance(texts, (list, tuple)): texts = [texts]
        embeddings = embedding_model.encode(texts, convert_to_tensor=False)
        embeddings_list = [emb.tolist() for emb in embeddings]
        logging.debug(f"Generated {len(embeddings_list)} embeddings.")
        return embeddings_list
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}", exc_info=True); return None

def check_similarity(candidate_embedding, existing_embeddings, threshold):
    """Checks if a candidate embedding is too similar to any existing embeddings."""
    if candidate_embedding is None or not existing_embeddings: return False
    try:
        candidate_tensor = np.array([candidate_embedding])
        existing_tensors = np.array(existing_embeddings)
        if candidate_tensor.ndim == 1: candidate_tensor = candidate_tensor[np.newaxis, :]
        if existing_tensors.ndim == 1: existing_tensors = existing_tensors[np.newaxis, :]
        if existing_tensors.shape[0] == 0: return False
        cosine_scores = util.cos_sim(candidate_tensor, existing_tensors)[0]
        max_similarity = 0.0
        if isinstance(cosine_scores, np.ndarray) and cosine_scores.size > 0: max_similarity = np.max(cosine_scores)
        elif torch is not None and isinstance(cosine_scores, torch.Tensor) and cosine_scores.numel() > 0: max_similarity = torch.max(cosine_scores).item()
        elif isinstance(cosine_scores, (list, tuple)) and len(cosine_scores) > 0: max_similarity = max(cosine_scores)
        logging.debug(f"Max similarity: {max_similarity:.4f}")
        return max_similarity > threshold
    except Exception as e:
        logging.error(f"Error checking similarity: {e}", exc_info=True); return False

# --- Trend Analysis Functions ---

def get_promising_ngrams(idea_names, top_n=3):
    """Analyzes idea names using N-grams and returns top themes."""
    if not idea_names: return []
    logging.info(f"Analyzing {len(idea_names)} ideas for N-gram trends...")
    all_features = []
    for name in idea_names:
        tokens = tokenize_and_clean(name)
        if tokens:
            all_features.extend(tokens)
            all_features.extend(generate_ngrams(tokens, 2))
            all_features.extend(generate_ngrams(tokens, 3))
    if not all_features: logging.warning("N-gram: No meaningful features extracted."); return []
    counts = Counter(all_features)
    filtered = {f: c for f, c in counts.items() if c > 1}
    if not filtered: logging.warning("N-gram: No features appeared more than once."); filtered = counts
    sorted_features = sorted(filtered.items(), key=lambda item: item[1], reverse=True)
    themes = [f for f, c in sorted_features[:top_n]]
    logging.info(f"N-gram Themes: {themes}")
    return themes

def get_lda_topics(idea_names, num_topics=3, num_words=3):
    """Performs LDA topic modeling on idea names."""
    if not SKLEARN_AVAILABLE: logging.warning("LDA disabled: scikit-learn not installed."); return []
    if not idea_names: return []
    logging.info(f"Analyzing {len(idea_names)} ideas for LDA topics...")
    try:
        # Preprocess text for LDA (simple join of tokens)
        processed_docs = [" ".join(tokenize_and_clean(name)) for name in idea_names if tokenize_and_clean(name)]
        if len(processed_docs) < num_topics: # Need enough documents
             logging.warning(f"LDA: Not enough valid documents ({len(processed_docs)}) for {num_topics} topics."); return []

        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        dtm = vectorizer.fit_transform(processed_docs)
        feature_names = vectorizer.get_feature_names_out()

        lda = LatentDirichletAllocation(n_components=num_topics, max_iter=10,
                                        learning_method='online', random_state=42)
        lda.fit(dtm)

        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_features_ind = topic.argsort()[:-num_words - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            topics.append(top_features) # List of lists of words

        # Flatten and unique words from top topics for prompt
        lda_themes = list(set([word for sublist in topics for word in sublist]))
        logging.info(f"LDA Themes (Keywords): {lda_themes}")
        return lda_themes
    except Exception as e:
        logging.error(f"Error during LDA analysis: {e}", exc_info=True)
        return []

def get_cluster_themes(idea_embeddings, idea_names, num_clusters=3, themes_per_cluster=1):
    """Performs K-Means clustering on embeddings and extracts themes from largest clusters."""
    if not SKLEARN_AVAILABLE: logging.warning("Clustering disabled: scikit-learn not installed."); return []
    if not idea_embeddings or len(idea_embeddings) < num_clusters:
        logging.warning(f"Clustering: Not enough embeddings ({len(idea_embeddings)}) for {num_clusters} clusters."); return []
    logging.info(f"Analyzing {len(idea_embeddings)} embeddings using K-Means clustering (k={num_clusters})...")
    try:
        embeddings_array = np.array(idea_embeddings)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10) # Explicitly set n_init
        kmeans.fit(embeddings_array)
        labels = kmeans.labels_

        cluster_ideas = {i: [] for i in range(num_clusters)}
        for i, label in enumerate(labels):
            cluster_ideas[label].append(idea_names[i]) # Store original names per cluster

        # Get cluster sizes
        cluster_sizes = Counter(labels)
        logging.debug(f"Cluster sizes: {cluster_sizes}")

        # Analyze largest clusters (e.g., top 2 or all with > X members)
        # For simplicity, let's just take themes from all generated clusters for now
        cluster_themes = []
        for i in range(num_clusters):
            if i in cluster_ideas and cluster_ideas[i]:
                 # Extract N-grams from the ideas within this cluster
                 cluster_ngrams = get_promising_ngrams(cluster_ideas[i], top_n=themes_per_cluster)
                 cluster_themes.extend(cluster_ngrams)

        # Return unique themes from the clusters
        unique_themes = list(set(cluster_themes))
        logging.info(f"Clustering Themes (N-grams): {unique_themes}")
        return unique_themes
    except Exception as e:
        logging.error(f"Error during embedding clustering: {e}", exc_info=True)
        return []

def get_combined_themes(high_rated_ideas_data):
    """Combines themes from N-grams, LDA, and Clustering."""
    if not high_rated_ideas_data or len(high_rated_ideas_data) < config.TREND_ANALYSIS_MIN_IDEAS:
        logging.info(f"Not enough high-rated ideas ({len(high_rated_ideas_data)}/{config.TREND_ANALYSIS_MIN_IDEAS}) for combined trend analysis.")
        return []

    logging.info("Performing combined trend analysis (N-grams, LDA, Clustering)...")
    idea_names = [item['name'] for item in high_rated_ideas_data]
    # Assume embeddings need to be generated or retrieved here if not passed in
    # For now, let's assume we need to generate them (less efficient)
    # A better approach would be to retrieve stored embeddings from DB
    idea_embeddings = generate_embeddings(idea_names)

    ngram_themes = get_promising_ngrams(idea_names, top_n=config.TREND_NGRAM_COUNT)
    lda_themes = get_lda_topics(idea_names, num_topics=config.TREND_LDA_TOPICS, num_words=config.TREND_LDA_WORDS)
    cluster_themes = get_cluster_themes(idea_embeddings, idea_names, num_clusters=config.TREND_CLUSTER_COUNT, themes_per_cluster=config.TREND_CLUSTER_THEMES_PER_CLUSTER)

    # Combine and unique themes
    combined = set(ngram_themes) | set(lda_themes) | set(cluster_themes)
    final_themes = list(combined)

    logging.info(f"Combined unique promising themes: {final_themes}")
    return final_themes
