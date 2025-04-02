import re
import logging
from collections import Counter
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import json
import string
import config
try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    logging.warning("scikit-learn library not found. LDA and Clustering analysis will be disabled.")
    SKLEARN_AVAILABLE = False

# --- NLTK Data Download (Run once on import) ---
NLTK_DATA_LOADED = False
try:
    logging.info("Checking/downloading NLTK data ('stopwords')...")
    try: nltk.data.find('corpora/stopwords')
    except LookupError: logging.info("Downloading NLTK 'stopwords'..."); nltk.download('stopwords', quiet=True)
    nltk.data.find('corpora/stopwords') # Verify again
    NLTK_DATA_LOADED = True
    logging.info("NLTK 'stopwords' data loaded successfully.")
except Exception as nltk_init_exc:
     logging.error(f"Failed to download/verify NLTK 'stopwords': {nltk_init_exc}.")
     NLTK_DATA_LOADED = False

# --- Global Variable for Embedding Model ---
embedding_model = None
try:
    device_to_use = 'cpu'
    logging.info(f"Loading sentence transformer model: {config.EMBEDDING_MODEL} onto device: {device_to_use}...")
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL, device=device_to_use)
    logging.info("Sentence transformer model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load sentence transformer model '{config.EMBEDDING_MODEL}': {e}", exc_info=True)

# --- Constants ---
if NLTK_DATA_LOADED: NLTK_STOPWORDS = set(stopwords.words('english'))
else: NLTK_STOPWORDS = set()
CUSTOM_STOPWORDS = set([ # Keep custom stop words
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
PUNCTUATION_TABLE = str.maketrans('', '', string.punctuation)

# --- Functions ---

def tokenize_and_clean(text):
    """Tokenizes text using simple split, converts to lowercase, removes punctuation, stop words and short words."""
    if not text: return []
    try:
        text_lower = text.lower().translate(PUNCTUATION_TABLE)
        words = text_lower.split()
        return [word for word in words if word not in ALL_STOP_WORDS and len(word) > 2]
    except Exception as e:
        logging.error(f"Error tokenizing text: {e}"); return []

def generate_ngrams(tokens, n):
    """Generates n-grams from a list of tokens."""
    try:
        from nltk.util import ngrams
        if len(tokens) < n: return []
        return [" ".join(gram) for gram in ngrams(tokens, n)]
    except ImportError:
         logging.warning("nltk.util.ngrams not found, using basic n-gram generation.")
         if len(tokens) < n: return []
         return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

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
        processed_docs = [" ".join(tokenize_and_clean(name)) for name in idea_names if tokenize_and_clean(name)]
        # Need at least as many documents as topics for LDA
        if len(processed_docs) < num_topics:
             logging.warning(f"LDA: Not enough valid documents ({len(processed_docs)}) for {num_topics} topics."); return []

        # Lower min_df to 1 to be less strict with small sample sizes
        vectorizer = CountVectorizer(max_df=0.95, min_df=1, stop_words='english')
        dtm = vectorizer.fit_transform(processed_docs)
        if dtm.shape[1] == 0:
             logging.warning("LDA: Vocabulary is empty after vectorization."); return []

        feature_names = vectorizer.get_feature_names_out()
        actual_num_topics = min(num_topics, dtm.shape[1])
        if actual_num_topics != num_topics: logging.warning(f"LDA: Reduced topics to {actual_num_topics}.")
        if actual_num_topics == 0: return []

        lda = LatentDirichletAllocation(n_components=actual_num_topics, max_iter=10, learning_method='online', random_state=42)
        lda.fit(dtm)
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            words_to_get = min(num_words, len(feature_names))
            top_features_ind = topic.argsort()[:-words_to_get - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            topics.append(top_features)
        lda_themes = list(set([word for sublist in topics for word in sublist]))
        logging.info(f"LDA Themes (Keywords): {lda_themes}")
        return lda_themes
    except Exception as e:
        logging.error(f"Error during LDA analysis: {e}", exc_info=True); return []

def get_cluster_themes(idea_embeddings, idea_names, num_clusters=3, themes_per_cluster=1):
    """Performs K-Means clustering on embeddings and extracts themes from largest clusters."""
    if not SKLEARN_AVAILABLE: logging.warning("Clustering disabled: scikit-learn not installed."); return []
    if not idea_embeddings or len(idea_embeddings) < num_clusters:
        logging.warning(f"Clustering: Not enough embeddings ({len(idea_embeddings)}) for {num_clusters} clusters."); return []
    logging.info(f"Analyzing {len(idea_embeddings)} embeddings using K-Means clustering (k={num_clusters})...")
    try:
        embeddings_array = np.array(idea_embeddings)
        actual_num_clusters = min(num_clusters, len(idea_embeddings))
        if actual_num_clusters != num_clusters: logging.warning(f"Clustering: Reduced clusters to {actual_num_clusters}.")

        kmeans = KMeans(n_clusters=actual_num_clusters, random_state=42, n_init=10)
        kmeans.fit(embeddings_array)
        labels = kmeans.labels_
        cluster_ideas = {i: [] for i in range(actual_num_clusters)}
        for i, label in enumerate(labels): cluster_ideas[label].append(idea_names[i])
        cluster_sizes = Counter(labels); logging.debug(f"Cluster sizes: {cluster_sizes}")
        cluster_themes = []
        for i in range(actual_num_clusters):
            if i in cluster_ideas and cluster_ideas[i]:
                 cluster_ngrams = get_promising_ngrams(cluster_ideas[i], top_n=themes_per_cluster)
                 cluster_themes.extend(cluster_ngrams)
        unique_themes = list(set(cluster_themes))
        logging.info(f"Clustering Themes (N-grams): {unique_themes}")
        return unique_themes
    except Exception as e:
        logging.error(f"Error during embedding clustering: {e}", exc_info=True); return []

def get_combined_themes(high_rated_ideas_data):
    """Combines themes from N-grams, LDA, and Clustering."""
    if not high_rated_ideas_data or len(high_rated_ideas_data) < config.TREND_ANALYSIS_MIN_IDEAS:
        logging.info(f"Not enough high-rated ideas ({len(high_rated_ideas_data)}/{config.TREND_ANALYSIS_MIN_IDEAS}) for combined trend analysis.")
        return []

    logging.info("Performing combined trend analysis (N-grams, LDA, Clustering)...")
    idea_names = [item['name'] for item in high_rated_ideas_data]
    idea_embeddings = generate_embeddings(idea_names)

    ngram_themes = get_promising_ngrams(idea_names, top_n=config.TREND_NGRAM_COUNT)
    lda_themes = get_lda_topics(idea_names, num_topics=config.TREND_LDA_TOPICS, num_words=config.TREND_LDA_WORDS)
    cluster_themes = get_cluster_themes(idea_embeddings, idea_names, num_clusters=config.TREND_CLUSTER_COUNT, themes_per_cluster=config.TREND_CLUSTER_THEMES_PER_CLUSTER)

    combined = set(ngram_themes) | set(lda_themes) | set(cluster_themes)
    final_themes = list(combined)
    logging.info(f"Combined unique promising themes: {final_themes}")
    return final_themes

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
