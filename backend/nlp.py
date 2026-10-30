"""TF-IDF / cosine baseline for answer evaluation + K-Means topic clusters.

LIMITATIONS (also shown to users): this is lexical matching. It sees shared words, not meaning:
a wrong answer that reuses the right vocabulary can score well, and a correct answer worded
differently (synonyms, paraphrase) can score low. Scores are a study aid, not a verdict.
"""
import re
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STOP = set(ENGLISH_STOP_WORDS) | {"example", "used", "use", "using", "also", "can", "allows", "like"}


def stem(w):
    for suf in ("ations", "ation", "ings", "ing", "ies", "es", "ed", "ly", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def tokens(text):
    out = []
    for w in re.findall(r"[a-z0-9+#.\-]+", text.lower().replace("/", " ")):
        w = w.strip(".-")
        if len(w) > 1 and w not in STOP:
            w = stem(w)
            if w not in STOP:
                out.append(w)
    return out


def _bigrams(toks):
    return set(zip(toks, toks[1:]))


class Evaluator:
    def __init__(self, reference_texts):
        self.vec = TfidfVectorizer(analyzer=tokens, sublinear_tf=True).fit(reference_texts)
        self.idf = dict(zip(self.vec.get_feature_names_out(), self.vec.idf_))

    def key_concepts(self, reference, n=8):
        toks = list(dict.fromkeys(tokens(reference)))
        toks.sort(key=lambda t: -self.idf.get(t, 1.0))
        return toks[:n]

    def evaluate(self, question, reference, answer):
        a_tok = set(tokens(answer))
        concepts = self.key_concepts(reference)
        covered = [c for c in concepts if c in a_tok]
        missing = [c for c in concepts if c not in a_tok]
        w = {c: self.idf.get(c, 1.0) for c in concepts}
        coverage = sum(w[c] for c in covered) / max(sum(w.values()), 1e-9)
        v = self.vec.transform([answer, reference, question + " " + reference])
        sim = float(cosine_similarity(v[0], v[1])[0, 0])
        rel = float(cosine_similarity(v[0], v[2])[0, 0])
        sim_s, rel_s = min(1, sim / 0.6), min(1, rel / 0.5)
        rb, ab = _bigrams(tokens(reference)), _bigrams(tokens(answer))
        if rb:  # shared two-word phrases (e.g. "gradient descent") reward correct terminology and word order
            acc_s = 0.7 * sim_s + 0.3 * min(1, (len(rb & ab) / len(rb)) / 0.5)
        else:
            acc_s = sim_s
        score = 100 * (0.45 * coverage + 0.35 * acc_s + 0.20 * rel_s)
        n_words, ref_words, notes = len(answer.split()), len(reference.split()), []
        if n_words < 4:
            score = min(score, 25)
            notes.append("Very short answer - explain in at least a full sentence.")
        elif ref_words >= 15 and n_words < ref_words * 0.4:
            score = min(score, 70)
            notes.append("Your answer is much shorter than a full answer would be; add detail and an example.")
        score = round(max(0, min(100, score)))
        label = "Strong" if score >= 75 else "Partial" if score >= 45 else "Needs work"
        return dict(score=score, label=label,
                    breakdown=dict(technical_accuracy=round(100 * acc_s), relevance=round(100 * rel_s),
                                   completeness=round(100 * coverage)),
                    covered_concepts=covered, missing_concepts=missing, notes=notes)


def add_clusters(df, k=14):
    ok = df["status"] == "converted"
    texts = (df.loc[ok, "question"] + " " + df.loc[ok, "answer"]).tolist()
    vec = TfidfVectorizer(analyzer=tokens, sublinear_tf=True, min_df=2)
    X = vec.fit_transform(texts)
    km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(X)
    terms = np.array(vec.get_feature_names_out())
    labels = {i: ", ".join(terms[np.argsort(-km.cluster_centers_[i])[:3]]) for i in range(k)}
    df["cluster_id"], df["cluster_label"] = -1, ""
    df.loc[ok, "cluster_id"] = km.labels_
    df.loc[ok, "cluster_label"] = [labels[i] for i in km.labels_]
    return df
