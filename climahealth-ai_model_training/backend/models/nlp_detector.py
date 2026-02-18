"""
ClimaHealth AI — NLP Outbreak Signal Detection Module
======================================================
Text classification pipeline for detecting disease outbreak signals
from news articles and public health reports.

Pipeline:
1. TF-IDF vectorization of news headlines/articles
2. Binary classification: outbreak-related vs. non-outbreak
3. Severity scoring based on keyword matching + model confidence
4. Entity extraction for disease type and location

In production, this would use:
- Fine-tuned BERT or Claude API for higher accuracy
- GDELT API for real-time global news feeds
- ProMED-mail RSS for curated outbreak reports
- WHO Disease Outbreak News API
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, f1_score, roc_auc_score
import joblib
import os
import re


# Disease keywords for entity extraction
DISEASE_KEYWORDS = {
    "dengue": ["dengue", "aedes", "hemorrhagic fever", "breakbone"],
    "malaria": ["malaria", "anopheles", "plasmodium", "antimalarial", "artemisinin"],
    "cholera": ["cholera", "vibrio", "watery diarrhea", "oral rehydration", "waterborne"],
    "zika": ["zika", "microcephaly", "guillain-barré", "guillain-barre"],
    "lyme": ["lyme", "borrelia", "tick-borne", "ixodes"],
}

# Severity indicator keywords
SEVERITY_KEYWORDS = {
    "critical": ["emergency", "crisis", "epidemic", "pandemic", "outbreak kills",
                  "overwhelmed", "deadly", "death toll", "surge", "catastrophic"],
    "high": ["outbreak", "spike", "surge", "increase", "spreading", "alarm",
             "hospitalizations", "cases rise", "detected"],
    "medium": ["elevated", "monitoring", "risk", "identified", "reported",
               "surveillance", "warning"],
}

# Location extraction patterns
LOCATION_KEYWORDS = {
    "dhaka_bangladesh": ["dhaka", "bangladesh", "bangla"],
    "nairobi_kenya": ["nairobi", "kenya", "kenyan"],
    "recife_brazil": ["recife", "pernambuco"],
    "chittagong_bangladesh": ["chittagong", "chattogram", "rohingya", "cox's bazar"],
    "lagos_nigeria": ["lagos", "nigeria", "nigerian"],
    "manaus_brazil": ["manaus", "amazonas", "amazon"],
}


class OutbreakSignalDetector:
    """
    NLP pipeline for detecting disease outbreak signals in text.
    Combines TF-IDF classification with rule-based entity extraction.
    """
    
    def __init__(self):
        self.pipeline = None
        self.tfidf = None
        self.is_fitted = False
    
    def fit(self, news_df):
        """
        Train the outbreak detection classifier.
        
        Args:
            news_df: DataFrame with 'headline' and 'is_outbreak' columns
        """
        print("\n  Training NLP Outbreak Signal Detector...")
        
        X = news_df["headline"].values
        y = news_df["is_outbreak"].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # TF-IDF + Logistic Regression pipeline
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 3),    # Unigrams, bigrams, and trigrams
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
                stop_words="english",
            )),
            ("clf", LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ))
        ])
        
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        y_prob = self.pipeline.predict_proba(X_test)[:, 1]
        
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        print(f"\n  === NLP Classification Results ===")
        print(f"  F1 Score: {f1:.3f}")
        print(f"  AUC-ROC:  {auc:.3f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Non-Outbreak', 'Outbreak'])}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.pipeline, X_train, y_train, cv=5, scoring="f1")
        print(f"  5-Fold CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Store the TF-IDF vectorizer for feature analysis
        self.tfidf = self.pipeline.named_steps["tfidf"]
        self.is_fitted = True
        
        return {"f1": f1, "auc": auc, "cv_f1_mean": cv_scores.mean()}
    
    def predict(self, texts):
        """
        Classify texts as outbreak-related or not.
        
        Args:
            texts: list of strings (headlines or article snippets)
            
        Returns:
            list of dicts with classification results
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        
        if isinstance(texts, str):
            texts = [texts]
        
        probs = self.pipeline.predict_proba(texts)[:, 1]
        preds = self.pipeline.predict(texts)
        
        results = []
        for text, prob, pred in zip(texts, probs, preds):
            result = {
                "text": text,
                "is_outbreak": bool(pred),
                "confidence": round(float(prob), 3),
                "disease": self._extract_disease(text),
                "severity": self._assess_severity(text, prob),
                "location": self._extract_location(text),
            }
            results.append(result)
        
        return results
    
    def compute_signal_score(self, texts):
        """
        Compute an aggregate NLP signal score from multiple texts.
        Returns a 0-1 score representing the strength of outbreak signals.
        """
        if not texts:
            return 0.0
        
        results = self.predict(texts)
        
        # Weighted score: outbreak texts contribute more, severity boosts score
        severity_weights = {"critical": 1.5, "high": 1.2, "medium": 1.0, "low": 0.5}
        
        total_score = 0
        for r in results:
            if r["is_outbreak"]:
                weight = severity_weights.get(r["severity"], 1.0)
                total_score += r["confidence"] * weight
        
        # Normalize to 0-1
        max_possible = len(texts) * 1.5  # Maximum if all critical outbreaks
        signal = min(1.0, total_score / max_possible)
        
        return round(signal, 3)
    
    def _extract_disease(self, text):
        """Extract mentioned disease from text using keyword matching."""
        text_lower = text.lower()
        for disease, keywords in DISEASE_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    return disease
        return "unknown"
    
    def _assess_severity(self, text, confidence):
        """Assess the severity of an outbreak signal."""
        text_lower = text.lower()
        
        for severity, keywords in SEVERITY_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    return severity
        
        # Fall back to confidence-based severity
        if confidence > 0.85:
            return "high"
        elif confidence > 0.6:
            return "medium"
        return "low"
    
    def _extract_location(self, text):
        """Extract location from text using keyword matching."""
        text_lower = text.lower()
        for region, keywords in LOCATION_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    return region
        return "unknown"
    
    def get_top_features(self, n=20):
        """Get the most important TF-IDF features for outbreak classification."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        
        clf = self.pipeline.named_steps["clf"]
        feature_names = self.tfidf.get_feature_names_out()
        
        # Get coefficients (positive = outbreak, negative = non-outbreak)
        coefs = clf.coef_[0]
        
        # Top outbreak indicators
        top_outbreak_idx = np.argsort(coefs)[-n:][::-1]
        outbreak_features = [(feature_names[i], round(coefs[i], 3)) for i in top_outbreak_idx]
        
        # Top non-outbreak indicators
        top_non_idx = np.argsort(coefs)[:n]
        non_features = [(feature_names[i], round(coefs[i], 3)) for i in top_non_idx]
        
        return {
            "outbreak_indicators": outbreak_features,
            "non_outbreak_indicators": non_features,
        }
    
    def save(self, path):
        """Save the NLP pipeline to disk."""
        os.makedirs(path, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline}, os.path.join(path, "nlp_detector.pkl"))
        print(f"  NLP model saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load the NLP pipeline from disk."""
        data = joblib.load(os.path.join(path, "nlp_detector.pkl"))
        obj = cls()
        obj.pipeline = data["pipeline"]
        obj.tfidf = obj.pipeline.named_steps["tfidf"]
        obj.is_fitted = True
        return obj
