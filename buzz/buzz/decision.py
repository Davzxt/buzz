"""Caminho de decisão rápida: TF-IDF de caracteres + regressão logística com temperatura calibrada.
Classifica a intenção em microssegundos, sem gerar tokens. Confiança != verdade."""
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Decision:
    def __init__(self, vec=None, clf=None, temp=1.0):
        self.vec, self.clf, self.temp = vec, clf, temp

    def fit(self, train, val):
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, strip_accents="unicode")
        X = self.vec.fit_transform([r["text"] for r in train])
        self.clf = LogisticRegression(C=10, max_iter=2000).fit(X, [r["label"] for r in train])
        # calibração por temperatura em templates NÃO vistos no treino
        zv = self.clf.decision_function(self.vec.transform([r["text"] for r in val]))
        yv = np.array([list(self.clf.classes_).index(r["label"]) for r in val])
        best = min(np.arange(0.05, 8.01, 0.05),
                   key=lambda t: -np.log(_softmax(zv / t)[np.arange(len(yv)), yv] + 1e-9).mean())
        self.temp = float(best)
        return self

    def proba(self, texts):
        return _softmax(self.clf.decision_function(self.vec.transform(texts)) / self.temp)

    def predict(self, text):
        p = self.proba([text])[0]
        i = int(p.argmax())
        return self.clf.classes_[i], float(p[i])

    def save(self, path):
        joblib.dump({"vec": self.vec, "clf": self.clf, "temp": self.temp}, path, compress=3)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        return cls(d["vec"], d["clf"], d["temp"])
