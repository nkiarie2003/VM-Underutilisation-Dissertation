# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split


# ============================================================
# FILE PATHS
# ============================================================

# Find the main project folder.
project_folder = Path(__file__).resolve().parent.parent

data_folder = project_folder / "data"

# Change this filename only if your final VM dataset has
# a different name.
input_file = data_folder / "reference_labels.csv"

# Files created by this experiment.
summary_output_file = (
    data_folder / "feedback_comparison_summary.csv"
)

all_runs_output_file = (
    data_folder / "feedback_comparison_all_runs.csv"
)


# ============================================================
# SETTINGS
# ============================================================

# These are the seeds used to repeat the whole experiment.
SEEDS = [
    42, 7, 21, 100, 123,
    1, 2, 3, 5, 8,
    13, 55, 77, 99, 200,
    314, 512, 777, 1000, 2024
]
# 20% of all VMs are kept completely separate for final testing.
TEST_SIZE = 0.20

# From the remaining development data, 30% is placed into
# the feedback pool.
FEEDBACK_POOL_SIZE = 0.30

# Number of VMs reviewed during each feedback cycle.
REVIEW_SIZE = 20

# Number of feedback cycles after the initial model.
N_CYCLES = 5

# Random Forest settings.
N_ESTIMATORS = 300


# ============================================================
# FEATURES
# ============================================================

# These must match the features used in Experiment A.

FEATURES = [
    "cpu_mean",
    "cpu_last_3_days_mean",
    "cpu_p95",
    "cpu_max",
    "cpu_below_5_pct",
    "memory_mean",
    "disk_mean",
    "outbound_network_mean",
    "network_mean",
    "cpu_cores",
]

TARGET = "reference_label"


# ============================================================
# LOAD DATA
# ============================================================

vm_data = pd.read_csv(
    input_file
)

print(
    "Loaded",
    len(vm_data),
    "virtual machines."
)

print("\nDataset:")
print("VMs:", len(vm_data))
print("Features:", len(FEATURES))

print("\nReference-label distribution:")
print(
    vm_data[TARGET].value_counts()
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = FEATURES + [TARGET]

missing_columns = [
    column
    for column in required_columns
    if column not in vm_data.columns
]

if missing_columns:

    raise ValueError(
        "Missing columns: "
        + ", ".join(missing_columns)
    )


# ============================================================
# MODEL FUNCTION
# ============================================================

def create_model(seed):
    """
    Create a fresh Random Forest.

    A new model is created every cycle so that the model is
    genuinely retrained after reviewed VMs are added.
    """

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=seed,
        n_jobs=-1,
    )

    return model


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
):
    """
    Evaluate the model using the untouched test set.
    """

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    # Confidence is the probability assigned to the
    # model's predicted class.
    confidence = np.max(
        probabilities,
        axis=1,
    )

    mean_confidence = np.mean(
        confidence
    )

    return {
        "accuracy_pct": accuracy * 100,
        "precision_pct": precision * 100,
        "recall_pct": recall * 100,
        "f1_pct": f1 * 100,
        "mean_confidence_pct": mean_confidence * 100,
    }


# ============================================================
# FEEDBACK EXPERIMENT FUNCTION
# ============================================================

def run_feedback_method(
    X_initial,
    y_initial,
    X_pool,
    y_pool,
    X_test,
    y_test,
    method,
    seed,
):
    """
    Run one complete five-cycle feedback experiment.

    method can be:

    "confidence"
        Select the least-confident VMs.

    "random"
        Select random VMs.

    The final test set is never used for selecting feedback.
    """

    # Make copies so the original split is not changed.
    X_training = X_initial.copy()
    y_training = y_initial.copy()

    X_feedback = X_pool.copy()
    y_feedback = y_pool.copy()

    results = []

    # Separate random generator for random selection.
    rng = np.random.default_rng(
        seed
    )

    # Cycle 0 is the model BEFORE any feedback is added.
    for cycle in range(
        N_CYCLES + 1
    ):

        model = create_model(
            seed
        )

        model.fit(
            X_training,
            y_training,
        )

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        result = {
            "seed": seed,
            "method": method,
            "cycle": cycle,
            "training_vms": len(X_training),
            "feedback_vms_remaining": len(X_feedback),
            **metrics,
        }

        results.append(
            result
        )

        # Stop after evaluating cycle 5.
        # We do not need to select another feedback batch.
        if cycle == N_CYCLES:
            break

        # If fewer than REVIEW_SIZE VMs remain,
        # use however many are left.
        number_to_review = min(
            REVIEW_SIZE,
            len(X_feedback),
        )

        # ----------------------------------------------------
        # CONFIDENCE-TARGETED SELECTION
        # ----------------------------------------------------

        if method == "confidence":

            pool_probabilities = (
                model.predict_proba(
                    X_feedback
                )
            )

            pool_confidence = np.max(
                pool_probabilities,
                axis=1,
            )

            # Get the positions of the least-confident VMs.
            selected_positions = np.argsort(
                pool_confidence
            )[:number_to_review]

        # ----------------------------------------------------
        # RANDOM CONTROL SELECTION
        # ----------------------------------------------------

        elif method == "random":

            selected_positions = rng.choice(
                len(X_feedback),
                size=number_to_review,
                replace=False,
            )

        else:

            raise ValueError(
                "Method must be 'confidence' or 'random'."
            )

        # Get the actual dataframe indexes for selected VMs.
        selected_indexes = (
            X_feedback.iloc[
                selected_positions
            ].index
        )

        # Add reviewed VMs to the training data.
        X_training = pd.concat(
            [
                X_training,
                X_feedback.loc[
                    selected_indexes
                ],
            ]
        )

        y_training = pd.concat(
            [
                y_training,
                y_feedback.loc[
                    selected_indexes
                ],
            ]
        )

        # Remove reviewed VMs from the feedback pool.
        X_feedback = X_feedback.drop(
            selected_indexes
        )

        y_feedback = y_feedback.drop(
            selected_indexes
        )

    return results


# ============================================================
# RUN EXPERIMENTS
# ============================================================

all_results = []


for seed in SEEDS:

    print(
        "\n"
        + "=" * 60
    )

    print(
        "RANDOM SEED:",
        seed,
    )

    print(
        "=" * 60
    )

    X = vm_data[
        FEATURES
    ].copy()

    y = vm_data[
        TARGET
    ].copy()


    # ========================================================
    # SPLIT 1:
    # DEVELOPMENT DATA + UNTOUCHED FINAL TEST SET
    # ========================================================

    (
        X_development,
        X_test,
        y_development,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=y,
    )


    # ========================================================
    # SPLIT 2:
    # INITIAL TRAINING DATA + FEEDBACK POOL
    # ========================================================

    (
        X_initial,
        X_pool,
        y_initial,
        y_pool,
    ) = train_test_split(
        X_development,
        y_development,
        test_size=FEEDBACK_POOL_SIZE,
        random_state=seed,
        stratify=y_development,
    )


    print(
        "\nInitial training VMs:",
        len(X_initial),
    )

    print(
        "Feedback-pool VMs:",
        len(X_pool),
    )

    print(
        "Untouched final-test VMs:",
        len(X_test),
    )


    # ========================================================
    # METHOD 1:
    # CONFIDENCE-TARGETED FEEDBACK
    # ========================================================

    confidence_results = (
        run_feedback_method(
            X_initial=X_initial,
            y_initial=y_initial,
            X_pool=X_pool,
            y_pool=y_pool,
            X_test=X_test,
            y_test=y_test,
            method="confidence",
            seed=seed,
        )
    )

    all_results.extend(
        confidence_results
    )


    # ========================================================
    # METHOD 2:
    # RANDOM-SELECTION CONTROL
    # ========================================================

    random_results = (
        run_feedback_method(
            X_initial=X_initial,
            y_initial=y_initial,
            X_pool=X_pool,
            y_pool=y_pool,
            X_test=X_test,
            y_test=y_test,
            method="random",
            seed=seed,
        )
    )

    all_results.extend(
        random_results
    )


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    all_results
)


# ============================================================
# SUMMARY ACROSS SEEDS
# ============================================================

summary_df = (
    results_df
    .groupby(
        [
            "method",
            "cycle",
        ]
    )
    .agg(
        mean_accuracy_pct=(
            "accuracy_pct",
            "mean",
        ),

        sd_accuracy_pct=(
            "accuracy_pct",
            "std",
        ),

        mean_precision_pct=(
            "precision_pct",
            "mean",
        ),

        sd_precision_pct=(
            "precision_pct",
            "std",
        ),

        mean_recall_pct=(
            "recall_pct",
            "mean",
        ),

        sd_recall_pct=(
            "recall_pct",
            "std",
        ),

        mean_f1_pct=(
            "f1_pct",
            "mean",
        ),

        sd_f1_pct=(
            "f1_pct",
            "std",
        ),

        mean_confidence_pct=(
            "mean_confidence_pct",
            "mean",
        ),

        sd_confidence_pct=(
            "mean_confidence_pct",
            "std",
        ),
    )
    .reset_index()
)


# Round values so the terminal output is easier to read.
numeric_columns = summary_df.select_dtypes(
    include="number"
).columns

summary_df[
    numeric_columns
] = summary_df[
    numeric_columns
].round(2)


# ============================================================
# PRINT F1 COMPARISON
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "CONFIDENCE-TARGETED VS RANDOM FEEDBACK"
)

print(
    "=" * 70
)


f1_table = summary_df.pivot(
    index="cycle",
    columns="method",
    values="mean_f1_pct",
)

print(
    "\nMean F1 across",
    len(SEEDS),
    "seeds:"
)

print(
    f1_table.round(2)
)


# ============================================================
# PRINT MEAN ± STANDARD DEVIATION
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "F1 MEAN +/- STANDARD DEVIATION"
)

print(
    "=" * 70
)


for method in [
    "confidence",
    "random",
]:

    print(
        "\nMethod:",
        method.upper(),
    )

    method_results = summary_df[
        summary_df["method"] == method
    ]

    for _, row in method_results.iterrows():

        print(
            "Cycle",
            int(row["cycle"]),
            ":",
            round(
                row["mean_f1_pct"],
                2,
            ),
            "+/-",
            round(
                row["sd_f1_pct"],
                2,
            ),
            "%"
        )


# ============================================================
# CALCULATE OVERALL CHANGE
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "OVERALL CHANGE"
)

print(
    "=" * 70
)


for method in [
    "confidence",
    "random",
]:

    method_results = summary_df[
        summary_df["method"] == method
    ]

    cycle_0 = method_results[
        method_results["cycle"] == 0
    ]["mean_f1_pct"].iloc[0]

    cycle_5 = method_results[
        method_results["cycle"] == 5
    ]["mean_f1_pct"].iloc[0]

    change = (
        cycle_5
        - cycle_0
    )

    print(
        "\n",
        method.upper(),
        sep="",
    )

    print(
        "Cycle 0 mean F1:",
        round(cycle_0, 2),
        "%"
    )

    print(
        "Cycle 5 mean F1:",
        round(cycle_5, 2),
        "%"
    )

    print(
        "F1 change:",
        round(change, 2),
        "percentage points"
    )


# ============================================================
# DIRECT CYCLE 5 COMPARISON
# ============================================================

confidence_cycle_5 = summary_df[
    (
        summary_df["method"]
        == "confidence"
    )
    &
    (
        summary_df["cycle"]
        == 5
    )
]["mean_f1_pct"].iloc[0]


random_cycle_5 = summary_df[
    (
        summary_df["method"]
        == "random"
    )
    &
    (
        summary_df["cycle"]
        == 5
    )
]["mean_f1_pct"].iloc[0]


difference = (
    confidence_cycle_5
    - random_cycle_5
)


print(
    "\n"
    + "=" * 70
)

print(
    "FINAL COMPARISON"
)

print(
    "=" * 70
)

print(
    "Confidence-targeted Cycle 5 F1:",
    round(
        confidence_cycle_5,
        2,
    ),
    "%"
)

print(
    "Random-selection Cycle 5 F1:",
    round(
        random_cycle_5,
        2,
    ),
    "%"
)

print(
    "Difference:",
    round(
        difference,
        2,
    ),
    "percentage points"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    all_runs_output_file,
    index=False,
)

summary_df.to_csv(
    summary_output_file,
    index=False,
)


print(
    "\nAll individual runs written to:"
)

print(
    all_runs_output_file
)

print(
    "\nSummary written to:"
)

print(
    summary_output_file
)

print(
    "\nExperiment complete."
)

# PAIRED SIGNIFICANCE TEST
# ============================================================

from scipy.stats import ttest_rel, wilcoxon

# Compare confidence-targeted and random selection at Cycle 5.
# Each value is paired by random seed.

cycle_5_results = results_df[
    results_df["cycle"] == 5
]

confidence_f1 = (
    cycle_5_results[
        cycle_5_results["method"] == "confidence"
    ]
    .sort_values("seed")["f1_pct"]
    .to_numpy()
)

random_f1 = (
    cycle_5_results[
        cycle_5_results["method"] == "random"
    ]
    .sort_values("seed")["f1_pct"]
    .to_numpy()
)

differences = confidence_f1 - random_f1

print("\n" + "=" * 70)
print("PAIRED SIGNIFICANCE TEST - CYCLE 5")
print("=" * 70)

print("\nConfidence-targeted F1 by seed:")
print(confidence_f1)

print("\nRandom-selection F1 by seed:")
print(random_f1)

print("\nPaired differences (confidence - random):")
print(differences)

print(
    "\nMean paired difference:",
    round(differences.mean(), 3),
    "percentage points"
)

# Paired t-test
t_stat, t_p = ttest_rel(
    confidence_f1,
    random_f1
)

print("\nPaired t-test")
print("t-statistic:", round(t_stat, 4))
print("p-value:", round(t_p, 4))

# Wilcoxon signed-rank test
# Useful as an additional non-parametric check because
# only five paired seeds are available.
try:
    w_stat, w_p = wilcoxon(
        confidence_f1,
        random_f1
    )

    print("\nWilcoxon signed-rank test")
    print("W-statistic:", round(w_stat, 4))
    print("p-value:", round(w_p, 4))

except ValueError as error:
    print("\nWilcoxon test could not be calculated:")
    print(error)

print("\nInterpretation at alpha = 0.05:")

if t_p < 0.05:
    print(
        "The paired t-test indicates a statistically significant "
        "difference between the two selection methods."
    )
else:
    print(
        "The paired t-test does not indicate a statistically significant "
        "difference between the two selection methods."
    )

print(
    "\nNote: the experiment used 20 paired random seeds. "
    "Statistical significance should still be interpreted alongside "
    "the small absolute effect size and high starting performance."
)