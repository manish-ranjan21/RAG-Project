import re

import config


class Guardrails:
    MIN_LEN = 5
    MAX_LEN = 500
    RELEVANCE_THRESHOLD = config.RELEVANCE_THRESHOLD

    TOPIC_KEYWORDS = {
        "ai",
        "ml",
        "machine learning",
        "deep learning",
        "neural",
        "model",
        "training",
        "dataset",
        "algorithm",
        "classification",
        "regression",
        "transformer",
        "llm",
        "embedding",
        "gradient",
        "backprop",
        "gan",
        "cnn",
        "rnn",
        "lstm",
        "attention",
        "reinforcement",
        "supervised",
        "unsupervised",
        "inference",
        "prediction",
        "feature",
        "overfitting",
    }

    INJECTION_PATTERNS = [
        r"ignore (previous|above|all) instructions",
        r"you are now",
        r"forget (everything|your instructions)",
        r"system prompt",
        r"act as (a )?(different|new|another)",
        r"disregard",
        r"jailbreak",
    ]

    def check_input(self, query: str) -> tuple[bool, str]:
        q = query.strip()
        if len(q) < self.MIN_LEN:
            return False, "Query too short. Please ask a full question."
        if len(q) > self.MAX_LEN:
            return False, f"Query too long (max {self.MAX_LEN} chars)."
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, q, re.IGNORECASE):
                return False, "Query contains disallowed instructions."
        return True, ""

    def check_topic(self, query: str) -> tuple[bool, str]:
        if any(kw in query.lower() for kw in self.TOPIC_KEYWORDS):
            return True, ""
        return False, "Warning: query may be outside the scope of the loaded books (AI/ML topics)."

    def check_relevance(self, avg_score: float) -> tuple[bool, str]:
        if avg_score > self.RELEVANCE_THRESHOLD:
            return (
                False,
                "Retrieved chunks have low relevance. Answer may not be grounded in the documents.",
            )
        return True, ""
