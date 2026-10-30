"""Train a supervised classifier that predicts interview-question difficulty (Easy / Medium / Hard).

Pipeline: processed dataset -> clean text -> stratified train/test split (random_state=42) -> TF-IDF fitted on the
TRAIN split only (no leakage) -> Logistic Regression (class_weight='balanced') -> metrics on the held-out test split.
Everything reported is computed here; nothing is hard-coded. Run:  python train_model.py
"""
import json, re, sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "unified_questions.csv"
MODEL_DIR = ROOT / "models"
LABELS = ["Easy", "Medium", "Hard"]
SEED = 42
PREFIX = re.compile(r"^complete the statement and explain:\s*", re.I)


def preprocess(text):
    t = PREFIX.sub("", str(text)).lower()
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"[^a-z0-9+#./\- ]", " ", t)          # keep technical symbols such as c++, c#, node.js, ci/cd
    return re.sub(r"\s+", " ", t).strip()


def make_vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def make_model():
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED)


def load_data(path=DATA):
    df = pd.read_csv(path).fillna("")
    df = df[~df["status"].isin(["duplicate", "invalid"])]                 # duplicates removed before splitting
    df = df[df["difficulty"].isin(LABELS) & (df["question"].str.strip() != "")].copy()   # drop missing labels / empty text
    df["text"] = df["question"].map(preprocess)
    return df.drop_duplicates("text").reset_index(drop=True)


def train(out_dir=MODEL_DIR, data_path=DATA):
    df = load_data(data_path)
    X_tr, X_te, y_tr, y_te, i_tr, i_te = train_test_split(df["text"], df["difficulty"], df.index, test_size=0.2,
                                                           stratify=df["difficulty"], random_state=SEED)
    vec = make_vectorizer().fit(X_tr)                  # fitted on training data only
    model = make_model().fit(vec.transform(X_tr), y_tr)
    proba = model.predict_proba(vec.transform(X_te))
    pred = model.classes_[proba.argmax(1)]
    acc = accuracy_score(y_te, pred)
    p, r, f, s = precision_recall_fscore_support(y_te, pred, labels=LABELS, zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y_te, pred, labels=LABELS, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_te, pred, labels=LABELS, average="weighted", zero_division=0)
    cm = confusion_matrix(y_te, pred, labels=LABELS)
    maj = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    base_acc = accuracy_score(y_te, maj.predict(X_te))
    base_f1 = precision_recall_fscore_support(y_te, maj.predict(X_te), labels=LABELS, average="macro", zero_division=0)[2]
    cv = cross_val_score(make_pipeline(make_vectorizer(), make_model()), df["text"], df["difficulty"],
                         cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="f1_macro")
    names, coef = np.array(vec.get_feature_names_out()), model.coef_
    top = {c: names[np.argsort(-coef[i])[:8]].tolist() for i, c in enumerate(model.classes_)}
    order = np.random.RandomState(SEED).permutation(len(X_te))[:10]
    samples = [dict(question=df.loc[i_te[j], "question"], actual=y_te.iloc[j], predicted=pred[j],
                    confidence=round(float(proba[j].max()), 3)) for j in order]
    lift = acc - base_acc
    if fm >= 0.6:
        verdict = "The model separates difficulty levels reasonably well from wording alone."
    elif fm > base_f1 + 0.1 and lift > 0:
        verdict = ("The model is better than the majority-class baseline, but only moderately accurate. "
                   "Difficulty depends on more than wording, and the labels are not guaranteed to be consistent.")
    else:
        verdict = ("The model is barely better than guessing the most common label. Question wording alone does not "
                   "predict this dataset's difficulty labels well, so treat predictions as a rough hint only.")
    metrics = dict(model_name="TF-IDF (1-2 grams) + Logistic Regression (class_weight=balanced)", random_state=SEED,
                   dataset_size=int(len(df)), train_size=int(len(X_tr)), test_size=int(len(X_te)), split="80% train / 20% test (stratified)",
                   classes=LABELS, class_counts=df["difficulty"].value_counts().to_dict(),
                   accuracy=round(acc, 4), precision_macro=round(pm, 4), recall_macro=round(rm, 4), f1_macro=round(fm, 4),
                   precision_weighted=round(pw, 4), recall_weighted=round(rw, 4), f1_weighted=round(fw, 4),
                   per_class={c: dict(precision=round(p[i], 4), recall=round(r[i], 4), f1=round(f[i], 4), support=int(s[i])) for i, c in enumerate(LABELS)},
                   confusion_matrix=cm.tolist(), baseline_majority_accuracy=round(base_acc, 4), baseline_majority_f1_macro=round(base_f1, 4),
                   cv_f1_macro_mean=round(float(cv.mean()), 4), cv_f1_macro_std=round(float(cv.std()), 4), vocabulary_size=int(len(names)),
                   top_terms_per_class=top, sample_predictions=samples, verdict=verdict,
                   limitations=["Labels come from the dataset and may not reflect true interview difficulty.",
                                "Only question wording is used, not topic knowledge or answer length.",
                                "Many questions follow templates, so test scores can be optimistic compared with brand-new questions."])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, out_dir / "tfidf_vectorizer.joblib")
    joblib.dump(model, out_dir / "difficulty_model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    m = train()
    print(m["model_name"], "| n =", m["dataset_size"], "| train/test =", m["train_size"], m["test_size"])
    print("accuracy", m["accuracy"], "macro P/R/F1", m["precision_macro"], m["recall_macro"], m["f1_macro"])
    print("baseline acc", m["baseline_majority_accuracy"], "baseline F1", m["baseline_majority_f1_macro"], "| 5-fold CV F1", m["cv_f1_macro_mean"], "+/-", m["cv_f1_macro_std"])
    print("confusion", m["confusion_matrix"]); print(m["verdict"])
