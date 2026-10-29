import os
import re
import sqlite3
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "project.csv"
)

DATABASE_PATH = os.environ.get("DATABASE_PATH") or os.path.join(
    BASE_DIR,
    "projectiq.db"
)

# Minimum similarity score (15%) to be listed as a similar project
SIMILARITY_THRESHOLD = 0.15


# ============================================================
# DOMAIN TAXONOMY & NORMALIZATION
# ============================================================

DOMAIN_FAMILIES = {
    "AI_ML": [
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "computer vision",
        "natural language processing",
        "nlp",
        "data science",
        "ai/ml",
        "ai",
        "ml",
        "generative ai"
    ],
    "WEB_DEV": [
        "web development",
        "web application",
        "web",
        "full stack",
        "frontend",
        "backend",
        "website",
        "portal"
    ],
    "MOBILE": [
        "mobile application",
        "mobile app",
        "android",
        "ios",
        "flutter",
        "react native",
        "mobile"
    ],
    "IOT": [
        "iot",
        "internet of things",
        "embedded systems",
        "embedded",
        "robotics",
        "sensors",
        "arduino",
        "raspberry pi",
        "smart"
    ],
    "CYBERSECURITY": [
        "cybersecurity",
        "security",
        "ethical hacking",
        "network security",
        "information security",
        "cryptography",
        "malware",
        "threat"
    ],
    "BLOCKCHAIN": [
        "blockchain",
        "web3",
        "smart contracts",
        "ethereum",
        "crypto",
        "decentralized"
    ],
    "CLOUD": [
        "cloud computing",
        "cloud",
        "devops",
        "docker",
        "kubernetes",
        "aws",
        "microservices"
    ]
}


def normalize_domain(domain_str):
    """
    Map any user-selected or dataset domain string to a canonical domain family.
    """
    if not domain_str:
        return "OTHER"

    domain_clean = clean_text(domain_str).lower()

    for family, aliases in DOMAIN_FAMILIES.items():
        for alias in aliases:
            if alias in domain_clean or domain_clean in alias:
                return family

    return "OTHER"


def are_domains_aligned(domain1, domain2):
    """
    Check if two domains belong to the same family or share high relevance.
    """
    fam1 = normalize_domain(domain1)
    fam2 = normalize_domain(domain2)

    if fam1 == "OTHER" or fam2 == "OTHER":
        return fam1 == fam2 and fam1 != "OTHER"

    return fam1 == fam2


# ============================================================
# LOAD DATASET AND DATABASE PROJECTS
# ============================================================

def load_dataset():
    """
    Load project dataset from datasets/project.csv
    """
    try:
        if not os.path.exists(DATASET_PATH):
            print("Dataset not found:", DATASET_PATH)
            return pd.DataFrame(columns=["title", "description", "domain", "source"])

        df = pd.read_csv(DATASET_PATH)

        required_columns = ["title", "description", "domain"]
        for column in required_columns:
            if column not in df.columns:
                df[column] = ""

        df = df[required_columns].fillna("")
        df["source"] = "Benchmark Repository"

        return df

    except Exception as e:
        print("Dataset loading error:", e)
        return pd.DataFrame(columns=["title", "description", "domain", "source"])


def load_database_projects(exclude_project_id=None):
    """
    Load previously evaluated projects from SQLite database.
    This guarantees that past student submissions are also checked for duplication!
    """
    try:
        if not os.path.exists(DATABASE_PATH):
            return pd.DataFrame(columns=["title", "description", "domain", "source"])

        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        query = "SELECT id, title, description, domain FROM projects"
        params = ()
        if exclude_project_id:
            query += " WHERE id != ?"
            params = (exclude_project_id,)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return pd.DataFrame(columns=["title", "description", "domain", "source"])

        records = []
        for r in rows:
            records.append({
                "title": r[1] or "",
                "description": r[2] or "",
                "domain": r[3] or "",
                "source": "Previous Submission"
            })

        return pd.DataFrame(records)

    except Exception as e:
        print("Database projects loading error:", e)
        return pd.DataFrame(columns=["title", "description", "domain", "source"])


def load_all_reference_projects(exclude_project_id=None, include_database=True):
    """
    Combine both the baseline dataset and previous submissions into a unified reference pool.
    """
    csv_df = load_dataset()

    if include_database:
        db_df = load_database_projects(exclude_project_id=exclude_project_id)
        if not db_df.empty:
            combined = pd.concat([csv_df, db_df], ignore_index=True)
            # Deduplicate if exact same title
            combined = combined.drop_duplicates(subset=["title"], keep="first")
            return combined

    return csv_df


# ============================================================
# TEXT CLEANING & TOKENIZATION
# ============================================================

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
    "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves", "project", "system", "based",
    "using", "application", "portal"
}


def clean_text(text):
    """
    Clean text before similarity calculation.
    """
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_meaningful_tokens(text):
    """
    Extract meaningful content tokens (ignoring stop words and 1-letter tokens).
    """
    text = clean_text(text)
    if not text:
        return set()

    words = text.split()
    return {
        word for word in words
        if len(word) > 2 and word not in STOP_WORDS
    }


def overlap_coefficient(text1, text2):
    """
    Calculate the overlap (Szymkiewicz-Simpson) coefficient:
    Overlap = |A ∩ B| / min(|A|, |B|)
    Unlike Jaccard, this does not harshly penalize when one text is 1 sentence
    and the other is a 3-paragraph detailed description!
    """
    set1 = get_meaningful_tokens(text1)
    set2 = get_meaningful_tokens(text2)

    if not set1 or not set2:
        return 0.0, []

    intersection = set1.intersection(set2)
    min_size = min(len(set1), len(set2))

    if min_size == 0:
        return 0.0, []

    score = len(intersection) / min_size
    return score, sorted(list(intersection))


def dice_similarity(text1, text2):
    """
    Sorensen-Dice coefficient between two texts.
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    """
    set1 = get_meaningful_tokens(text1)
    set2 = get_meaningful_tokens(text2)

    if not set1 or not set2:
        return 0.0

    intersection = set1.intersection(set2)
    total = len(set1) + len(set2)

    if total == 0:
        return 0.0

    return (2.0 * len(intersection)) / total


# ============================================================
# TF-IDF COSINE SIMILARITY
# ============================================================

def calculate_tfidf_similarity(query_text, corpus_texts):
    """
    Calculate TF-IDF cosine similarity between query text and a list of corpus texts.
    Uses unigram and bigram representation with sublinear TF.
    """
    if not query_text or not corpus_texts:
        return [0.0] * len(corpus_texts)

    all_texts = [query_text] + list(corpus_texts)

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True
        )

        vectors = vectorizer.fit_transform(all_texts)
        similarity_matrix = cosine_similarity(vectors[0:1], vectors[1:])
        return similarity_matrix[0].tolist()

    except Exception as e:
        print("TF-IDF similarity error:", e)
        return [0.0] * len(corpus_texts)


# ============================================================
# EVALUATE PROJECT
# ============================================================

def evaluate_project(
    title,
    description,
    domain="",
    users="",
    technologies="",
    include_db_projects=True,
    exclude_project_id=None
):
    """
    Compare the student's project against existing benchmark and past projects.

    Parameters:
        title (str): Project title (Crucial for identifying project concepts)
        description (str): Project description / problem statement
        domain (str): Project domain (e.g., Artificial Intelligence, IoT)
        users (str): Target users / stakeholders
        technologies (str): Tech stack utilized
        include_db_projects (bool): Whether to include past student submissions
        exclude_project_id (int): Project ID to exclude if re-evaluating existing

    Returns:
        uniqueness (float): 0.0 to 100.0 (Higher means more novel / unique)
        similar_projects (list[dict]): Ranked list of similar projects with scores and details
    """

    df = load_all_reference_projects(
        exclude_project_id=exclude_project_id,
        include_database=include_db_projects
    )

    if df.empty:
        print("Reference repository is empty.")
        return 100.0, []

    clean_user_title = clean_text(title)
    clean_user_desc = clean_text(description)
    clean_user_domain = clean_text(domain)
    clean_user_tech = clean_text(technologies)

    # Combined user representation
    user_full_text = " ".join([
        clean_user_title,
        clean_user_title,  # emphasize title
        clean_user_desc,
        clean_user_domain,
        clean_user_tech
    ])

    if not user_full_text.strip():
        return 0.0, []

    # Build reference text representations
    ref_descriptions = []
    ref_titles = []
    ref_full_texts = []

    for _, row in df.iterrows():
        ref_t = clean_text(row.get("title", ""))
        ref_d = clean_text(row.get("description", ""))
        ref_dom = clean_text(row.get("domain", ""))

        ref_titles.append(ref_t)
        ref_descriptions.append(ref_d)
        ref_full_texts.append(f"{ref_t} {ref_t} {ref_d} {ref_dom}")

    df = df.reset_index(drop=True)

    # 1. TF-IDF Cosine Similarity on full texts
    cosine_scores = calculate_tfidf_similarity(user_full_text, ref_full_texts)

    similar_projects = []

    for idx in range(len(df)):
        row = df.iloc[idx]
        ref_title = ref_titles[idx]
        ref_desc = ref_descriptions[idx]
        ref_domain = row.get("domain", "")
        ref_source = row.get("source", "Benchmark Repository")

        cos_score = cosine_scores[idx]

        # 2. Title-to-Title Overlap & Dice Score
        title_dice = dice_similarity(clean_user_title, ref_title)
        title_overlap, title_matching_words = overlap_coefficient(clean_user_title, ref_title)
        title_score = (title_dice * 0.5) + (title_overlap * 0.5)

        # 3. Description Concept Overlap
        desc_overlap, desc_matching_words = overlap_coefficient(
            f"{clean_user_title} {clean_user_desc}",
            f"{ref_title} {ref_desc}"
        )

        # 4. Domain Alignment Factor
        domain_aligned = are_domains_aligned(domain, ref_domain)
        exact_domain_match = (
            clean_user_domain
            and clean_text(ref_domain)
            and clean_user_domain == clean_text(ref_domain)
        )

        domain_factor = 0.0
        if exact_domain_match:
            domain_factor = 0.12
        elif domain_aligned:
            domain_factor = 0.08
        else:
            # Different domain: apply mild damping if not a strong tech match
            domain_factor = -0.05

        # 5. Multi-Signal Composite Similarity Score
        # If title matches very strongly (e.g. "Smart Waste Management" vs "Smart Waste Management System"),
        # similarity must be high (>= 80%)
        if title_score >= 0.70:
            composite_score = (title_score * 0.55) + (cos_score * 0.35) + domain_factor
        else:
            composite_score = (
                (cos_score * 0.50) +
                (title_score * 0.25) +
                (desc_overlap * 0.15) +
                domain_factor
            )

        # Non-linear boost for strong matching concepts
        # (e.g. if key keywords match and domains align)
        all_matched_words = sorted(list(set(title_matching_words + desc_matching_words)))

        if len(all_matched_words) >= 4 and domain_aligned:
            composite_score += 0.08

        # Clamp between 0.0 and 1.0
        composite_score = min(max(composite_score, 0.0), 1.0)
        similarity_percentage = round(composite_score * 100.0, 1)

        # Determine risk level based on similarity percentage
        if similarity_percentage >= 70.0:
            risk_level = "High Overlap (Potential Duplicate)"
            risk_color = "danger"
        elif similarity_percentage >= 40.0:
            risk_level = "Moderate Overlap (Common Concept)"
            risk_color = "warning"
        elif similarity_percentage >= 20.0:
            risk_level = "Low Overlap (Novel Variations)"
            risk_color = "info"
        else:
            risk_level = "Minimal Overlap"
            risk_color = "success"

        if composite_score >= SIMILARITY_THRESHOLD:
            similar_projects.append({
                "title": str(row.get("title", "")),
                "description": str(row.get("description", "")),
                "domain": str(row.get("domain", "")),
                "similarity": similarity_percentage,
                "matched_terms": all_matched_words[:6],
                "risk_level": risk_level,
                "risk_color": risk_color,
                "source": ref_source
            })

    # Sort descending by similarity
    similar_projects.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    # Highest similarity found
    if similar_projects:
        max_similarity = similar_projects[0]["similarity"]
    else:
        max_similarity = 0.0

    # Calculate calibrated uniqueness
    # 100% - max_similarity
    uniqueness = round(max(0.0, min(100.0, 100.0 - max_similarity)), 1)

    return uniqueness, similar_projects


# ============================================================
# KEYWORD HELPERS
# ============================================================

def contains_any(text, keywords):
    """
    Check whether any keyword exists in text.
    """
    text = clean_text(text)
    return any(
        keyword.lower() in text
        for keyword in keywords
    )


def count_keywords(text, keywords):
    """
    Count how many keywords are present.
    """
    text = clean_text(text)
    return sum(
        1
        for keyword in keywords
        if keyword.lower() in text
    )


# ============================================================
# PROJECT SCORE CALCULATION
# ============================================================

def calculate_project_scores(
    description,
    domain,
    users,
    technologies,
    uniqueness
):
    """
    Calculate:
        innovation
        feasibility
        resume value
        overall
    """

    description_text = clean_text(description)
    domain_text = clean_text(domain)
    users_text = clean_text(users)
    technologies_text = clean_text(technologies)

    full_text = " ".join([
        description_text,
        domain_text,
        users_text,
        technologies_text
    ])

    # --------------------------------------------------------
    # INNOVATION
    # --------------------------------------------------------

    innovation = 50

    innovation_keywords = [
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "computer vision",
        "natural language processing",
        "nlp",
        "generative ai",
        "chatbot",
        "prediction",
        "recommendation",
        "automation",
        "blockchain",
        "iot",
        "real time",
        "face recognition",
        "object detection",
        "fraud detection",
        "sentiment analysis",
        "smart",
        "optimization"
    ]

    innovation += count_keywords(
        full_text,
        innovation_keywords
    ) * 4

    advanced_keywords = [
        "personalized",
        "adaptive",
        "explainable",
        "edge ai",
        "real-time",
        "multimodal",
        "secure",
        "intelligent",
        "automated"
    ]

    innovation += count_keywords(
        full_text,
        advanced_keywords
    ) * 3

    # Low uniqueness reduces innovation score
    if uniqueness < 30:
        innovation -= 18
    elif uniqueness < 50:
        innovation -= 10
    elif uniqueness < 70:
        innovation -= 3
    elif uniqueness >= 85:
        innovation += 10

    innovation = max(10, min(100, innovation))

    # --------------------------------------------------------
    # FEASIBILITY
    # --------------------------------------------------------

    feasibility = 65

    feasibility_keywords = [
        "web",
        "website",
        "database",
        "mysql",
        "sqlite",
        "flask",
        "django",
        "spring",
        "api",
        "frontend",
        "backend",
        "react",
        "javascript",
        "python",
        "java",
        "crud",
        "authentication"
    ]

    feasibility += count_keywords(
        full_text,
        feasibility_keywords
    ) * 2

    complex_keywords = [
        "large language model",
        "llm",
        "distributed system",
        "quantum",
        "autonomous drone",
        "satellite",
        "hardware fabrication"
    ]

    feasibility -= count_keywords(
        full_text,
        complex_keywords
    ) * 4

    feasibility = max(15, min(100, feasibility))

    # --------------------------------------------------------
    # RESUME VALUE
    # --------------------------------------------------------

    resume_value = 55

    resume_keywords = [
        "machine learning",
        "deep learning",
        "artificial intelligence",
        "computer vision",
        "nlp",
        "react",
        "spring boot",
        "flask",
        "node",
        "api",
        "cloud",
        "docker",
        "deployment",
        "authentication",
        "real time",
        "dashboard",
        "analytics",
        "prediction",
        "recommendation",
        "automation"
    ]

    resume_value += count_keywords(
        full_text,
        resume_keywords
    ) * 3

    if uniqueness >= 80:
        resume_value += 10
    elif uniqueness >= 60:
        resume_value += 5
    elif uniqueness < 35:
        resume_value -= 10

    resume_value = max(15, min(100, resume_value))

    # --------------------------------------------------------
    # OVERALL
    # --------------------------------------------------------

    overall = round(
        (
            (innovation * 0.30)
            + (feasibility * 0.25)
            + (resume_value * 0.25)
            + (uniqueness * 0.20)
        )
    )

    return {
        "innovation": innovation,
        "feasibility": feasibility,
        "resume": resume_value,
        "overall": overall
    }


# ============================================================
# RECOMMENDATION GENERATOR
# ============================================================

def generate_recommendations(
    description,
    domain,
    uniqueness
):
    """
    Generate customized recommendations based on project domain and uniqueness level.
    """

    description_text = clean_text(description)
    domain_text = clean_text(domain)
    full_text = f"{description_text} {domain_text}"

    recommendations = []

    # Uniqueness Specific Recommendations
    if uniqueness < 35:
        recommendations.append(
            "High Similarity Alert: This concept is widely implemented. Add a distinct novel module (e.g., edge AI inference, real-time alerts, or hybrid architecture) to avoid rejection."
        )
        recommendations.append(
            "Differentiate your target audience: Solve the problem for a specialized sector or regional context rather than a generic template."
        )
    elif uniqueness < 65:
        recommendations.append(
            "Moderate Novelty: Enhance your project by integrating an advanced workflow like real-time streaming, automated report generation, or multi-role dashboards."
        )
    else:
        recommendations.append(
            "Strong Novelty: Your project concept has solid uniqueness! Ensure you thoroughly document the methodology and validate with benchmark metrics."
        )

    # Domain specific guidance
    fam = normalize_domain(domain)

    if fam == "AI_ML":
        if not contains_any(full_text, ["dataset", "kaggle", "data"]):
            recommendations.append(
                "Specify a high-quality real-world dataset and detail your data preprocessing, cleaning, and augmentation pipeline."
            )
        if not contains_any(full_text, ["accuracy", "precision", "recall", "f1", "confusion matrix", "evaluation"]):
            recommendations.append(
                "Include rigorous model evaluation metrics (Precision, Recall, F1-Score, ROC-AUC) and compare against baseline algorithms."
            )
        if not contains_any(full_text, ["api", "flask", "fastapi", "docker", "deploy"]):
            recommendations.append(
                "Deploy the trained model via a REST API (FastAPI / Flask) with a clean user-facing frontend to demonstrate production readiness."
            )
        if not contains_any(full_text, ["explainable", "shap", "lime", "interpret"]):
            recommendations.append(
                "Incorporate Explainable AI (XAI) using SHAP or LIME so evaluators can see how predictions are made."
            )

    elif fam == "WEB_DEV":
        if not contains_any(full_text, ["jwt", "oauth", "auth", "rbac", "role"]):
            recommendations.append(
                "Implement secure Role-Based Access Control (RBAC) with JWT or OAuth2 for proper access hierarchy."
            )
        if not contains_any(full_text, ["responsive", "tailwind", "mobile friendly"]):
            recommendations.append(
                "Ensure the web interface is fully responsive across mobile, tablet, and desktop viewports."
            )
        if not contains_any(full_text, ["cache", "redis", "indexing", "performance"]):
            recommendations.append(
                "Add database indexing or Redis caching to highlight high-throughput architecture in your viva presentation."
            )

    elif fam == "IOT":
        if not contains_any(full_text, ["mqtt", "websocket", "real-time"]):
            recommendations.append(
                "Use MQTT or WebSockets for lightweight, real-time telemetry streaming from sensor nodes to the cloud."
            )
        if not contains_any(full_text, ["sms", "email", "notification", "alert"]):
            recommendations.append(
                "Configure automated alert triggers (SMS / Telegram / Email) whenever critical sensor thresholds are breached."
            )
        if not contains_any(full_text, ["edge", "offline"]):
            recommendations.append(
                "Implement offline fallback or edge processing so the hardware node functions even during internet dropouts."
            )

    elif fam == "CYBERSECURITY":
        if not contains_any(full_text, ["mitre", "cve", "threat"]):
            recommendations.append(
                "Map your detection heuristics to the MITRE ATT&CK framework for standardized industry relevance."
            )
        if not contains_any(full_text, ["real time", "dashboard", "siem"]):
            recommendations.append(
                "Provide a live SOC security dashboard visualizing alerts, source IP geolocations, and attack severity."
            )

    elif fam == "BLOCKCHAIN":
        if not contains_any(full_text, ["gas", "audit", "security", "solidity"]):
            recommendations.append(
                "Include smart contract gas optimization analysis and automated vulnerability testing (Slither / Mythril)."
            )
        if not contains_any(full_text, ["ipfs", "storage"]):
            recommendations.append(
                "Integrate IPFS (InterPlanetary File System) for decentralized off-chain media and metadata storage."
            )

    # General recommendations
    if not contains_any(full_text, ["docker", "ci/cd", "github actions"]):
        recommendations.append(
            "Containerize the application with Docker and set up a CI/CD workflow to earn top marks for engineering quality."
        )

    # Deduplicate and return top 5
    unique_recs = []
    for r in recommendations:
        if r not in unique_recs:
            unique_recs.append(r)

    return unique_recs[:5]


# ============================================================
# CLI TEST RUNNER
# ============================================================

if __name__ == "__main__":
    test_title = "Crop Disease Detection using Computer Vision"
    test_description = "Machine learning system for detecting diseases in crops from leaf photos using deep learning CNN."
    test_domain = "Computer Vision"
    test_users = "Farmers and Agricultural Officers"
    test_technologies = "Python TensorFlow OpenCV Flask"

    uniqueness, similar_projects = evaluate_project(
        test_title,
        test_description,
        test_domain,
        test_users,
        test_technologies
    )

    print("\n================================")
    print("PROJECT IQ EVALUATION ENGINE TEST")
    print("================================")
    print(f"Test Project: {test_title}")
    print(f"Domain: {test_domain}")
    print(f"Uniqueness Score: {uniqueness}%\n")

    print(f"Top Similar Projects ({len(similar_projects)} found):")
    for p in similar_projects[:5]:
        print(f" - {p['title']} [{p['domain']}]: {p['similarity']}% ({p['risk_level']})")
        if p.get('matched_terms'):
            print(f"   Matched Terms: {', '.join(p['matched_terms'])}")

    scores = calculate_project_scores(
        test_description,
        test_domain,
        test_users,
        test_technologies,
        uniqueness
    )
    print(f"\nScores: {scores}")

    recs = generate_recommendations(test_description, test_domain, uniqueness)
    print("\nRecommendations:")
    for r in recs:
        print(f" * {r}")