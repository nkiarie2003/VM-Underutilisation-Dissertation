from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# 1. SETTINGS
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ESTIMATORS = 300

HIGH_CONFIDENCE = 0.90
MEDIUM_CONFIDENCE = 0.70


# ============================================================
# 2. FEATURES
# ============================================================

feature_columns = [
    "cpu_cores",
    "cpu_mean",
    "cpu_last_3_days_mean",
    "cpu_p95",
    "cpu_max",
    "cpu_below_5_pct",
    "memory_mean",
    "disk_mean",
    "outbound_network_mean",
    "network_mean",
]


# These features were also used when creating
# the frozen reference labels.

rule_features = [
    "cpu_mean",
    "cpu_last_3_days_mean",
    "cpu_p95",
    "cpu_max",
    "cpu_below_5_pct",
    "memory_mean",
    "disk_mean",
    "outbound_network_mean",
]


# ============================================================
# 3. LOAD DATA
# ============================================================

data_folder = (
    Path(__file__).parent.parent
    / "data"
)

reference_file = (
    data_folder
    / "reference_labels.csv"
)

baseline_file = (
    data_folder
    / "baseline_comparison.csv"
)


vm_data = pd.read_csv(
    reference_file
)

baseline_data = pd.read_csv(
    baseline_file
)


print(
    f"Loaded {len(vm_data)} virtual machines."
)


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = (
    feature_columns
    + [
        "vm_id",
        "reference_label",
    ]
)

missing_columns = [
    column
    for column in required_columns
    if column not in vm_data.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns from reference_labels.csv: "
        + ", ".join(missing_columns)
    )


baseline_required_columns = [
    "vm_id",
    "baseline_label",
]

baseline_missing_columns = [
    column
    for column in baseline_required_columns
    if column not in baseline_data.columns
]

if baseline_missing_columns:
    raise ValueError(
        "Missing required columns from baseline_comparison.csv: "
        + ", ".join(baseline_missing_columns)
    )


# ============================================================
# 5. CREATE X AND y
# ============================================================

X = vm_data[
    feature_columns
]

y = vm_data[
    "reference_label"
]


print("\nModel setup:")

print(
    "Number of input features:",
    X.shape[1],
)

print(
    "Number of VMs:",
    X.shape[0],
)


print("\nTarget distribution:")

print(
    y.value_counts()
)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
)


print("\nTrain/test split:")

print(
    "Training VMs:",
    len(X_train),
)

print(
    "Testing VMs:",
    len(X_test),
)


# ============================================================
# 7. AZURE-INSPIRED BASELINE
# SAME 243 TEST VMs AS RANDOM FOREST
# ============================================================

# Get the VM IDs that ended up in the Random Forest test set.

test_vm_ids = vm_data.loc[
    X_test.index,
    "vm_id"
].values


# Match baseline predictions to those exact same VMs.

baseline_test_data = (
    baseline_data
    .set_index("vm_id")
    .loc[test_vm_ids]
    .reset_index()
)


baseline_test = baseline_test_data[
    "baseline_label"
]


# Calculate baseline performance against the same
# reference labels used to evaluate the Random Forest.

baseline_accuracy = accuracy_score(
    y_test,
    baseline_test,
)

baseline_precision = precision_score(
    y_test,
    baseline_test,
    zero_division=0,
)

baseline_recall = recall_score(
    y_test,
    baseline_test,
    zero_division=0,
)

baseline_f1 = f1_score(
    y_test,
    baseline_test,
    zero_division=0,
)

baseline_matrix = confusion_matrix(
    y_test,
    baseline_test,
)


print("\n" + "=" * 60)

print(
    "AZURE-INSPIRED BASELINE - SAME 243 TEST VMs"
)

print("=" * 60)


print(
    "Accuracy:",
    round(
        baseline_accuracy * 100,
        1,
    ),
    "%",
)

print(
    "Precision:",
    round(
        baseline_precision * 100,
        1,
    ),
    "%",
)

print(
    "Recall:",
    round(
        baseline_recall * 100,
        1,
    ),
    "%",
)

print(
    "F1-score:",
    round(
        baseline_f1 * 100,
        1,
    ),
    "%",
)


print("\nConfusion matrix:")

print(
    baseline_matrix
)


# ============================================================
# 8. FUNCTION FOR RANDOM FOREST EXPERIMENTS
# ============================================================

def run_experiment(
    label,
    columns,
):

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )


    model.fit(
        X_train[columns],
        y_train,
    )


    y_pred = model.predict(
        X_test[columns]
    )


    prediction_probabilities = (
        model.predict_proba(
            X_test[columns]
        )
    )


    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        y_pred,
    )


    folds = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


    cv_scores = cross_val_score(

        RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),

        X[columns],

        y,

        cv=folds,

        scoring="f1",
    )


    print("\n" + "=" * 60)

    print(
        label
    )

    print("=" * 60)


    print(
        "Features used:",
        len(columns),
    )

    print(
        "Accuracy:",
        round(
            accuracy * 100,
            1,
        ),
        "%",
    )

    print(
        "Precision:",
        round(
            precision * 100,
            1,
        ),
        "%",
    )

    print(
        "Recall:",
        round(
            recall * 100,
            1,
        ),
        "%",
    )

    print(
        "F1-score:",
        round(
            f1 * 100,
            1,
        ),
        "%",
    )

    print(
        "5-fold CV F1:",
        round(
            cv_scores.mean() * 100,
            1,
        ),
        "%",
    )


    print("\nConfusion matrix:")

    print(
        matrix
    )


    return {
        "experiment": label,
        "n_features": len(columns),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cv_f1_mean": cv_scores.mean(),
        "cv_f1_sd": cv_scores.std(),
        "model": model,
        "columns": columns,
        "predictions": y_pred,
        "probabilities": prediction_probabilities,
    }


# ============================================================
# 9. EXPERIMENT A - MAIN RANDOM FOREST
# ============================================================

result_full = run_experiment(

    "EXPERIMENT A: Main Random Forest - all features",

    feature_columns,
)


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

print(
    "\nFeature importances:"
)


importances = pd.Series(

    result_full[
        "model"
    ].feature_importances_,

    index=feature_columns,

).sort_values(
    ascending=False
)


for name, value in importances.items():

    marker = ""

    if name in rule_features:
        marker = (
            " <-- used by reference-label rule"
        )

    print(
        f"{name:28s} "
        f"{value:.3f}"
        f"{marker}"
    )


# ============================================================
# 11. EXPERIMENT B - ABLATION
# ============================================================

ablated_columns = [

    column

    for column in feature_columns

    if column not in rule_features
]


print(
    "\nFeatures removed for ablation:"
)

print(
    ", ".join(
        rule_features
    )
)


print(
    "\nFeatures remaining:"
)

print(
    ", ".join(
        ablated_columns
    )
)


result_ablated = run_experiment(

    "EXPERIMENT B: Ablation - rule features removed",

    ablated_columns,
)


# ============================================================
# 12. MAJORITY CLASS BASELINE
# ============================================================

majority_class = (
    y_train
    .value_counts()
    .idxmax()
)


y_majority = np.full(

    shape=len(y_test),

    fill_value=majority_class,
)


majority_accuracy = accuracy_score(
    y_test,
    y_majority,
)


majority_f1 = f1_score(
    y_test,
    y_majority,
    zero_division=0,
)


print("\n" + "=" * 60)

print(
    "BASELINE: Majority class"
)

print("=" * 60)


print(
    "Predicted class for every VM:",
    majority_class,
)

print(
    "Accuracy:",
    round(
        majority_accuracy * 100,
        1,
    ),
    "%",
)

print(
    "F1-score:",
    round(
        majority_f1 * 100,
        1,
    ),
    "%",
)


# ============================================================
# 13. CONFIDENCE
# ============================================================

main_model = result_full[
    "model"
]

predictions = result_full[
    "predictions"
]

probabilities = result_full[
    "probabilities"
]


print(
    "\nRandom Forest class order:"
)

print(
    main_model.classes_
)


# Confidence = probability of whichever class
# the model selected.

confidence_scores = (
    probabilities.max(
        axis=1
    )
)


confidence_bands = np.where(

    confidence_scores
    >= HIGH_CONFIDENCE,

    "High",

    np.where(

        confidence_scores
        >= MEDIUM_CONFIDENCE,

        "Medium",

        "Low",
    ),
)


# ============================================================
# 14. RECOMMENDATIONS
# ============================================================

def recommendation_from_prediction(
    prediction,
    confidence_band,
):

    if not prediction:
        return "No Action"

    if confidence_band == "High":
        return "Shutdown"

    if confidence_band == "Medium":
        return "Resize"

    return "No Action"


recommendations = [

    recommendation_from_prediction(
        prediction,
        band,
    )

    for prediction, band
    in zip(
        predictions,
        confidence_bands,
    )
]


# ============================================================
# 15. CONFIDENCE RESULTS TABLE
# ============================================================

confidence_results = pd.DataFrame({

    "vm_id":
        vm_data.loc[
            X_test.index,
            "vm_id"
        ].values,

    "actual_label":
        y_test.values,

    "predicted_label":
        predictions,

    "probability_false":
        probabilities[:, 0],

    "probability_true":
        probabilities[:, 1],

    "confidence_score":
        confidence_scores,

    "confidence_band":
        confidence_bands,

    "recommendation":
        recommendations,
})


print(
    "\nFirst 10 confidence results:"
)


print(

    confidence_results
    .head(10)
    .round(3)
    .to_string(
        index=False
    )
)


print(
    "\nConfidence score summary:"
)

print(

    confidence_results[
        "confidence_score"
    ].describe()
)


print(
    "\nConfidence bands:"
)

print(

    confidence_results[
        "confidence_band"
    ].value_counts()
)


print(
    "\nRecommendations:"
)

print(

    confidence_results[
        "recommendation"
    ].value_counts()
)


# ============================================================
# 16. FINAL RESULTS TABLE
# ============================================================

summary = pd.DataFrame([

    {
        "experiment":
            "Azure-inspired threshold baseline",

        "n_features":
            0,

        "accuracy_pct":
            round(
                baseline_accuracy * 100,
                1,
            ),

        "precision_pct":
            round(
                baseline_precision * 100,
                1,
            ),

        "recall_pct":
            round(
                baseline_recall * 100,
                1,
            ),

        "f1_pct":
            round(
                baseline_f1 * 100,
                1,
            ),

        "cv_f1_pct":
            None,
    },

    {
        "experiment":
            "A: Main Random Forest",

        "n_features":
            result_full[
                "n_features"
            ],

        "accuracy_pct":
            round(
                result_full[
                    "accuracy"
                ] * 100,
                1,
            ),

        "precision_pct":
            round(
                result_full[
                    "precision"
                ] * 100,
                1,
            ),

        "recall_pct":
            round(
                result_full[
                    "recall"
                ] * 100,
                1,
            ),

        "f1_pct":
            round(
                result_full[
                    "f1"
                ] * 100,
                1,
            ),

        "cv_f1_pct":
            round(
                result_full[
                    "cv_f1_mean"
                ] * 100,
                1,
            ),
    },

    {
        "experiment":
            "B: Ablation",

        "n_features":
            result_ablated[
                "n_features"
            ],

        "accuracy_pct":
            round(
                result_ablated[
                    "accuracy"
                ] * 100,
                1,
            ),

        "precision_pct":
            round(
                result_ablated[
                    "precision"
                ] * 100,
                1,
            ),

        "recall_pct":
            round(
                result_ablated[
                    "recall"
                ] * 100,
                1,
            ),

        "f1_pct":
            round(
                result_ablated[
                    "f1"
                ] * 100,
                1,
            ),

        "cv_f1_pct":
            round(
                result_ablated[
                    "cv_f1_mean"
                ] * 100,
                1,
            ),
    },

    {
        "experiment":
            "Baseline: Majority class",

        "n_features":
            0,

        "accuracy_pct":
            round(
                majority_accuracy * 100,
                1,
            ),

        "precision_pct":
            0.0,

        "recall_pct":
            0.0,

        "f1_pct":
            round(
                majority_f1 * 100,
                1,
            ),

        "cv_f1_pct":
            None,
    },
])


print("\n" + "=" * 60)

print(
    "FINAL SUMMARY"
)

print("=" * 60)


print(

    summary.to_string(
        index=False
    )
)


# ============================================================
# 17. SAVE RESULTS
# ============================================================

results_file = (
    data_folder
    / "random_forest_results.csv"
)


confidence_file = (
    data_folder
    / "random_forest_confidence.csv"
)


summary.to_csv(
    results_file,
    index=False,
)


confidence_results.to_csv(
    confidence_file,
    index=False,
)


print(
    "\nModel summary written to:"
)

print(
    results_file
)


print(
    "\nConfidence results written to:"
)

print(
    confidence_file
)