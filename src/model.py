from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Mots neutres qui apportent peu d'informations
# pour la classification du sentiment.
# On conserve "ne" et "pas" pour les negations.
MOTS_NEUTRES = [
    "a", "achat", "au", "aux", "avec",
    "ce", "ces", "cette", "dans", "d",
    "de", "des", "du", "elle", "en",
    "est", "et", "experience", "il",
    "je", "l", "la", "le", "les",
    "livraison", "ma", "mes", "mon",
    "plus", "pour", "produit",
    "qualite", "que", "qui", "service",
    "sur", "tres", "un", "une",
]


def build_model(params):
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=params["model"]["max_features"],
                ngram_range=tuple(
                    params["model"]["ngram_range"]
                ),
                strip_accents="unicode",
                stop_words=MOTS_NEUTRES,
            ),
        ),
        (
            "clf",
            LogisticRegression(
                C=params["model"]["C"],
                max_iter=params["model"]["max_iter"],
                class_weight="balanced",
            ),
        ),
    ])
