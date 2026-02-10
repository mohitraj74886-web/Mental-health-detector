# =====================================
# 1. IMPORTS
# =====================================

import json
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

# =====================================
# 2. LOAD DATA
# =====================================

def load_json(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

train = load_json(r"C:\Users\vvkra\Downloads\train.json")
val   = load_json(r"C:\Users\vvkra\Downloads\val.json")
test  = load_json(r"C:\Users\vvkra\Downloads\test.json")

# =====================================
# 3. CLEAN TEXT
# =====================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

for df in [train, val, test]:
    df["clean_text"] = df["text"].apply(clean_text)


# =====================================
# 4. BINARY RISK LABEL
# =====================================

# HIGH RISK if suicide intent OR many emotions
def binary_risk(label):
    label = str(label)
    suicide = label[-1] == "1"
    ones = label.count("1")

    if suicide or ones >= 5:
        return 1
    return 0

for df in [train, val, test]:
    df["risk"] = df["label_id"].apply(binary_risk)

print("\nClass distribution:")
print(train["risk"].value_counts(normalize=True))


# =====================================
# 5. TF-IDF FEATURES
# =====================================

vectorizer = TfidfVectorizer(
    ngram_range=(1,3),
    max_features=20000,
    min_df=5,
    max_df=0.8,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(train["clean_text"])
X_val   = vectorizer.transform(val["clean_text"])
X_test  = vectorizer.transform(test["clean_text"])

y_train = train["risk"]
y_val   = val["risk"]
y_test  = test["risk"]


# =====================================
# 6. LOGISTIC REGRESSION GRIDSEARCH
# =====================================

lr = LogisticRegression(max_iter=2000)

lr_params = {"C":[0.1,1,5,10]}

lr_grid = GridSearchCV(
    lr,
    lr_params,
    scoring="f1",
    cv=3,
    n_jobs=-1
)

lr_grid.fit(X_train, y_train)

print("\nBest LR params:", lr_grid.best_params_)


# =====================================
# 7. SVM GRIDSEARCH
# =====================================

svm = LinearSVC()

svm_params = {
    "C":[0.01,0.1,1,5,10,20],
    "loss":["hinge","squared_hinge"],
    "max_iter":[3000,5000]
}

svm_grid = GridSearchCV(
    svm,
    svm_params,
    scoring="f1",
    cv=3,
    n_jobs=-1
)

svm_grid.fit(X_train, y_train)

print("Best SVM params:", svm_grid.best_params_)


# =====================================
# 8. VALIDATION COMPARISON
# =====================================

lr_val_pred  = lr_grid.predict(X_val)
svm_val_pred = svm_grid.predict(X_val)

lr_f1  = f1_score(y_val, lr_val_pred)
svm_f1 = f1_score(y_val, svm_val_pred)

print("\nValidation F1:")
print("Logistic:", lr_f1)
print("SVM:", svm_f1)

best_model = lr_grid if lr_f1 > svm_f1 else svm_grid


# =====================================
# 9. FINAL TEST EVALUATION
# =====================================

test_pred = best_model.predict(X_test)

print("\nFINAL TEST RESULTS:")
print(classification_report(y_test, test_pred))


# =====================================
# 10. CONFUSION MATRIX
# =====================================

ConfusionMatrixDisplay.from_predictions(y_test, test_pred)
plt.show()


# =====================================
# 11. EXPLAINABLE PREDICTION
# =====================================

def explain(text):
    vec = vectorizer.transform([clean_text(text)])
    pred = best_model.predict(vec)[0]

    print("\nPredicted Risk:",
          "HIGH" if pred==1 else "NOT HIGH")

    if hasattr(best_model.best_estimator_, "coef_"):
        coefs = best_model.best_estimator_.coef_[0]
        words = vectorizer.get_feature_names_out()

        top = np.argsort(coefs)[-10:]

        print("\nKey indicators:")
        for i in top:
            print(words[i])

import joblib

# Save best model
joblib.dump(best_model.best_estimator_, "svm_model.pkl")

# Save the single vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Saved successfully!")





