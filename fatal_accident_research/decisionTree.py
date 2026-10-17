import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

RANDOM_STATE = 42
pd.set_option("display.width", 140)

df = pd.read_excel("data/AccidentData21-26.csv.xlsx")
df.columns = df.columns.str.strip()

y = df["Crash Severity"].isin(["Fatal", "Suspected Serious Injury"]).astype(int)

features = [
    'Roadway Junction Type',
    'Light Condition',
    'Weather Condition',
    'Roadway Surface Condition',
    'Number of Vehicles Involved',
    'Route Type',
    'Intersection Involved',
    'Aggressive Driving Involved',
    'Bicycle Involved',
    'Collision With Fixed Object',
    'Commercial Motor Vehicle Involved',
    'Crash Type',
    'Disregard Traffic Control Device Involved',
    'Distracted Driving Involved',
    'Drowsy Driving Involved',
    'DUI Involved',
    'Estimated Travel Speed',
    'Hit and Run',
    'Manner of Collision',
    'Motorcycle Involved',
    'Pedestrian Involved',
    'Posted Speed',
    'Roadway Departure Crash',
    'Roadway Geometry Involved',
    'Roadway Type',
    'Speed Related',
    'Unrestrained Involved',
    'Work Zone Involved',
    'Wrong Way Driving Involved'
]
numeric_features = ['Number of Vehicles Involved', 'Estimated Travel Speed', 'Posted Speed']
categorical_features = [f for f in features if f not in numeric_features]

X = df[features].copy()
for col in numeric_features:
    X[col] = pd.to_numeric(X[col], errors="coerce")
X[categorical_features] = X[categorical_features].astype(object).fillna("Missing").astype(str)

print(f"{len(df)} crashes, {y.sum()} KA ({y.mean():.1%})")
print(f"Baseline accuracy from always predicting 'not KA': {1 - y.mean():.1%}\n")


def build_preprocess(scale):
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), categorical_features),
        ],
        verbose_feature_names_out=False,
    )


# SVM and ANN are distance/gradient based and need scaled inputs; the tree doesn't,
# and leaving it unscaled keeps its split thresholds in real units (e.g. mph).
models = {
    "Decision Tree": (DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=RANDOM_STATE), False),
    "SVM": (SVC(kernel="rbf", class_weight="balanced", cache_size=1000, random_state=RANDOM_STATE), True),
    "ANN": (MLPClassifier(hidden_layer_sizes=(32, 16), early_stopping=True, max_iter=500, random_state=RANDOM_STATE), True),
}


def make_pipeline_for(name):
    model, scale = models[name]
    return Pipeline([("prep", build_preprocess(scale)), ("model", clone(model))])


def ranking_scores(pipe, X_):
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X_)[:, 1]
    return pipe.decision_function(X_)


# --------------------------------------------------------------------------
# Compare models with 5-fold stratified CV on the training set
# --------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "recall": "recall",
    "precision": "precision",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}

results = {}
for name in models:
    print(f"Cross-validating {name}...")
    scores = cross_validate(make_pipeline_for(name), X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    results[name] = {metric: scores[f"test_{metric}"].mean() for metric in scoring}

comparison = pd.DataFrame(results).T.round(3).sort_values("pr_auc", ascending=False)
print("\nCross-validated model comparison (ranked by PR-AUC):")
print(comparison)

# --------------------------------------------------------------------------
# Evaluate the best model on the held-out test set
# --------------------------------------------------------------------------

best_name = comparison.index[0]
best_pipe = make_pipeline_for(best_name).fit(X_train, y_train)

test_scores = ranking_scores(best_pipe, X_test)
test_pred = best_pipe.predict(X_test)
print(f"\nBest model: {best_name}")
print(f"Test ROC-AUC: {roc_auc_score(y_test, test_scores):.3f}   Test PR-AUC: {average_precision_score(y_test, test_scores):.3f}")
print(classification_report(y_test, test_pred, target_names=["Not KA", "KA"], zero_division=0))
print("Confusion matrix (rows = actual, cols = predicted):")
print(pd.DataFrame(confusion_matrix(y_test, test_pred), index=["Not KA", "KA"], columns=["Not KA", "KA"]))

# --------------------------------------------------------------------------
# Which factors drive the best model's predictions?
# --------------------------------------------------------------------------

print(f"\nComputing permutation importance for {best_name}...")
perm = permutation_importance(
    best_pipe, X_test, y_test, scoring="average_precision",
    n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1,
)
importances = pd.Series(perm.importances_mean, index=features).sort_values(ascending=False)
print("\nTop 15 factors (drop in PR-AUC when the factor is shuffled):")
print(importances.head(15).round(4))

fig, ax = plt.subplots(figsize=(9, 6))
importances.head(15).plot(kind="barh", ax=ax)
ax.invert_yaxis()
ax.set_title(f"Most Predictive Factors of KA Crashes ({best_name})")
ax.set_xlabel("Drop in PR-AUC when shuffled")
plt.tight_layout()
fig.savefig("featureImportance.png", dpi=150)
plt.close(fig)

# Importance says how much a factor matters, not which direction; show KA rates for that.
print("\nKA rate by value for the top 5 factors:")
for feat in importances.head(5).index:
    groups = pd.qcut(X[feat], 5, duplicates="drop") if feat in numeric_features else X[feat]
    summary = y.groupby(groups, observed=True).agg(KA_rate="mean", crashes="size")
    summary = summary[summary["crashes"] >= 20].sort_values("KA_rate", ascending=False)
    summary["KA_rate"] = (summary["KA_rate"] * 100).round(1).astype(str) + "%"
    print(f"\n{feat}:")
    print(summary.head(8))

# --------------------------------------------------------------------------
# Decision tree diagram (always drawn, since it's the interpretable model)
# --------------------------------------------------------------------------

tree_pipe = make_pipeline_for("Decision Tree").fit(X_train, y_train)
plt.figure(figsize=(28, 14))
plot_tree(
    tree_pipe.named_steps["model"],
    feature_names=tree_pipe.named_steps["prep"].get_feature_names_out(),
    class_names=["Not KA", "KA"],
    filled=True,
    rounded=True,
    proportion=True,
    fontsize=9,
)
plt.savefig("crashFactorsTree.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nSaved featureImportance.png and crashFactorsTree.png")
