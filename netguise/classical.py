from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class GGFastVectorizer:
    """Length/direction n-gram counts for the GGFAST-style baseline."""

    def __init__(self, ngram: int = 2, length_bin: int = 64, max_packets: int = 50):
        self.ngram = ngram
        self.length_bin = length_bin
        self.max_packets = max_packets
        self.token_count = 2 * (1500 // length_bin + 1) + 1
        self.feature_count = self.token_count**ngram

    def transform(self, x: np.ndarray) -> np.ndarray:
        output = np.zeros((len(x), self.feature_count), dtype=np.float32)
        for row, sequence in enumerate(x[:, : self.max_packets]):
            valid = sequence[:, 0] > 0
            lengths = np.minimum(
                (sequence[valid, 0] // self.length_bin).astype(np.int64),
                1500 // self.length_bin,
            )
            directions = (sequence[valid, 1] < 0).astype(np.int64)
            tokens = lengths + directions * (1500 // self.length_bin + 1)
            for offset in range(max(0, len(tokens) - self.ngram + 1)):
                index = 0
                for token in tokens[offset : offset + self.ngram]:
                    index = index * self.token_count + int(token)
                output[row, index] += 1.0
        return output


class GradBPVectorizer:
    """Aggregate and cumulative early-flow statistics for an SVM baseline."""

    def __init__(self, max_packets: int = 30):
        self.max_packets = max_packets

    def transform(self, x: np.ndarray) -> np.ndarray:
        sequence = x[:, : self.max_packets]
        valid = sequence[..., 0] > 0
        count = valid.sum(axis=1).clip(min=1)
        sizes = sequence[..., 0]
        iats = sequence[..., 2]
        signed_sizes = sizes * sequence[..., 1]
        cumulative_size = np.cumsum(signed_sizes, axis=1) / 1500.0
        cumulative_time = np.cumsum(iats, axis=1)
        summaries = np.column_stack(
            [
                count,
                sizes.sum(axis=1),
                sizes.max(axis=1),
                sizes.sum(axis=1) / count,
                iats.sum(axis=1),
                iats.max(axis=1),
                (sequence[..., 1] > 0).sum(axis=1) / count,
            ]
        )
        return np.column_stack([summaries, cumulative_size, cumulative_time]).astype(np.float32)


@dataclass
class ClassicalClassifier:
    vectorizer: object
    estimator: object

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.estimator.predict(self.vectorizer.transform(x))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(self.vectorizer.transform(x))


def train_ggfast(x: np.ndarray, y: np.ndarray) -> ClassicalClassifier:
    from sklearn.naive_bayes import MultinomialNB

    vectorizer = GGFastVectorizer()
    estimator = MultinomialNB(alpha=0.5)
    estimator.fit(vectorizer.transform(x), y)
    return ClassicalClassifier(vectorizer, estimator)


def train_gradbp(x: np.ndarray, y: np.ndarray) -> ClassicalClassifier:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    vectorizer = GradBPVectorizer(max_packets=min(30, x.shape[1]))
    estimator = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(
            SVC(kernel="rbf", C=1.0, random_state=42),
            method="sigmoid",
            cv=3,
        ),
    )
    estimator.fit(vectorizer.transform(x), y)
    return ClassicalClassifier(vectorizer, estimator)
