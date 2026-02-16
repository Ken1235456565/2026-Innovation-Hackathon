"""
ClimaHealth AI — NLP Outbreak Signal Detection Module
======================================================
Fixed for Python 3.13 + pandas pyarrow backend compatibility.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, f1_score, roc_auc_score
import joblib
import os

DISEASE_KEYWORDS = {
    "dengue": ["dengue", "aedes", "hemorrhagic fever", "breakbone"],
    "malaria": ["malaria", "anopheles", "plasmodium", "antimalarial", "artemisinin"],
    "cholera": ["cholera", "vibrio", "watery diarrhea", "oral rehydration", "waterborne"],
    "zika": ["zika", "microcephaly", "guillain-barré", "guillain-barre"],
    "lyme": ["lyme", "borrelia", "tick-borne", "ixodes"],
}

SEVERITY_KEYWORDS = {
    "critical": ["emergency", "crisis", "epidemic", "pandemic", "outbreak kills",
                  "overwhelmed", "deadly", "death toll", "surge", "catastrophic"],
    "high": ["outbreak", "spike", "surge", "increase", "spreading", "alarm",
             "hospitalizations", "cases rise", "detected"],
    "medium": ["elevated", "monitoring", "risk", "identified", "reported",
               "surveillance", "warning"],
}

LOCATION_KEYWORDS = {
    "dhaka_bangladesh": ["dhaka", "bangladesh", "bangla"],
    "nairobi_kenya": ["nairobi", "kenya", "kenyan"],
    "recife_brazil": ["recife", "pernambuco"],
    "chittagong_bangladesh": ["chittagong", "chattogram", "rohingya", "cox's bazar"],
    "lagos_nigeria": ["lagos", "nigeria", "nigerian"],
    "manaus_brazil": ["manaus", "amazonas", "amazon"],
}


class OutbreakSignalDetector:
    def __init__(self):
        self.pipeline = None
        self.tfidf = None
        self.is_fitted = False

    def fit(self, news_df):
        print("\n  Training NLP Outbreak Signal Detector...")

        # CRITICAL FIX: Convert to numpy arrays to avoid pyarrow indexing issues
        X = np.array(news_df["headline"].tolist())
        y = np.array(news_df["is_outbreak"].tolist(), dtype=int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=5000, ngram_range=(1, 3), min_df=2,
                max_df=0.95, sublinear_tf=True, stop_words="english",
            )),
            ("clf", LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=1000, random_state=42,
            ))
        ])

        self.pipeline.fit(X_train, y_train)

        y_pred = self.pipeline.predict(X_test)
        y_prob = self.pipeline.predict_proba(X_test)[:, 1]
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        print(f"\n  === NLP Classification Results ===")
        print(f"  F1 Score: {f1:.3f}")
        print(f"  AUC-ROC:  {auc:.3f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Non-Outbreak', 'Outbreak'])}")

        cv_scores = cross_val_score(self.pipeline, X_train, y_train, cv=5, scoring="f1")
        print(f"  5-Fold CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        self.tfidf = self.pipeline.named_steps["tfidf"]
        self.is_fitted = True
        return {"f1": f1, "auc": auc, "cv_f1_mean": cv_scores.mean()}

    def predict(self, texts):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        if isinstance(texts, str):
            texts = [texts]
        probs = self.pipeline.predict_proba(texts)[:, 1]
        preds = self.pipeline.predict(texts)
        results = []
        for text, prob, pred in zip(texts, probs, preds):
            results.append({
                "text": text, "is_outbreak": bool(pred),
                "confidence": round(float(prob), 3),
                "disease": self._extract_disease(text),
                "severity": self._assess_severity(text, prob),
                "location": self._extract_location(text),
            })
        return results

    def compute_signal_score(self, texts):
        if not texts:
            return 0.0
        results = self.predict(texts)
        severity_weights = {"critical": 1.5, "high": 1.2, "medium": 1.0, "low": 0.5}
        total_score = sum(r["confidence"] * severity_weights.get(r["severity"], 1.0) for r in results if r["is_outbreak"])
        return round(min(1.0, total_score / (len(texts) * 1.5)), 3)

    def _extract_disease(self, text):
        text_lower = text.lower()
        for disease, keywords in DISEASE_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return disease
        return "unknown"

    def _assess_severity(self, text, confidence):
        text_lower = text.lower()
        for severity, keywords in SEVERITY_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return severity
        return "high" if confidence > 0.85 else "medium" if confidence > 0.6 else "low"

    def _extract_location(self, text):
        text_lower = text.lower()
        for region, keywords in LOCATION_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return region
        return "unknown"

    def get_top_features(self, n=20):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        clf = self.pipeline.named_steps["clf"]
        feature_names = self.tfidf.get_feature_names_out()
        coefs = clf.coef_[0]
        top_outbreak_idx = np.argsort(coefs)[-n:][::-1]
        outbreak_features = [(feature_names[i], round(coefs[i], 3)) for i in top_outbreak_idx]
        top_non_idx = np.argsort(coefs)[:n]
        non_features = [(feature_names[i], round(coefs[i], 3)) for i in top_non_idx]
        return {"outbreak_indicators": outbreak_features, "non_outbreak_indicators": non_features}

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline}, os.path.join(path, "nlp_detector.pkl"))
        print(f"  NLP model saved to {path}")

    @classmethod
    def load(cls, path):
        data = joblib.load(os.path.join(path, "nlp_detector.pkl"))
        obj = cls()
        obj.pipeline = data["pipeline"]
        obj.tfidf = obj.pipeline.named_steps["tfidf"]
        obj.is_fitted = True
        return obj
