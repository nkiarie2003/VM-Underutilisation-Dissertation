"""
Beyond Thresholds
Azure VM Underutilisation Decision-Support Prototype
"""

import os
import sqlite3
from pathlib import Path

import joblib
import pandas as pd

from flask import (
    Flask,
    Response,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import train_model



# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Original dissertation data/model packaged with the application
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATASET_PATH = DATA_DIR / "reference_labels.csv"
MODEL_PATH = MODEL_DIR / "random_forest.joblib"

# Runtime files
# Azure App Service uses /home for persistent writable storage.
# Locally, these continue to use the normal project folders.
if os.environ.get("WEBSITE_HOSTNAME"):
    RUNTIME_DIR = Path("/home/capacity-advisor")
else:
    RUNTIME_DIR = BASE_DIR

RUNTIME_DATA_DIR = RUNTIME_DIR / "data"
RUNTIME_MODEL_DIR = RUNTIME_DIR / "models"

FEEDBACK_PATH = RUNTIME_DATA_DIR / "feedback.csv"
DATABASE_PATH = RUNTIME_DATA_DIR / "reviews.db"
FEEDBACK_MODEL_PATH = RUNTIME_MODEL_DIR / "random_forest_feedback.joblib"

# Create writable runtime folders if they do not already exist
RUNTIME_DATA_DIR.mkdir(parents=True, exist_ok=True)
RUNTIME_MODEL_DIR.mkdir(parents=True, exist_ok=True)



# ============================================================
# Settings
# ============================================================

FEATURES = train_model.FEATURES

HIGH_CONFIDENCE = 0.90
MEDIUM_CONFIDENCE = 0.70


# ============================================================
# Flask
# ============================================================

app = Flask(__name__)

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "development-key",
)


# ============================================================
# Helpers
# ============================================================

def as_bool(value):

    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in {"true", "1", "yes"}:
        return True

    if text in {"false", "0", "no"}:
        return False

    raise ValueError(
        f"Could not convert {value} to boolean."
    )


def numeric_vm_id(series):

    numbers = (
        series.astype(str)
        .str.extract(r"(\d+)", expand=False)
    )

    return pd.to_numeric(
        numbers,
        errors="coerce",
    )


# ============================================================
# Review database
# ============================================================

def init_database():

    DATA_DIR.mkdir(exist_ok=True)

    with sqlite3.connect(DATABASE_PATH) as connection:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS reviews (
                vm_id TEXT PRIMARY KEY,
                model_prediction TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                confidence REAL NOT NULL,
                corrected_label TEXT NOT NULL,
                reviewer_notes TEXT,
                reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        connection.commit()


def save_review(
    vm_id,
    prediction,
    recommendation,
    confidence,
    corrected_label,
    reviewer_notes="",
):

    with sqlite3.connect(DATABASE_PATH) as connection:

        connection.execute(
            """
            INSERT OR REPLACE INTO reviews (
                vm_id,
                model_prediction,
                recommendation,
                confidence,
                corrected_label,
                reviewer_notes,
                reviewed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                str(vm_id),
                str(prediction),
                recommendation,
                float(confidence),
                corrected_label,
                reviewer_notes,
            ),
        )

        connection.commit()

    export_feedback()


def get_reviews():

    with sqlite3.connect(DATABASE_PATH) as connection:

        connection.row_factory = sqlite3.Row

        rows = connection.execute(
            """
            SELECT *
            FROM reviews
            ORDER BY reviewed_at DESC
            """
        ).fetchall()

    return [
        dict(row)
        for row in rows
    ]


def export_feedback():

    reviews = get_reviews()

    if not reviews:
        return

    dataframe = pd.DataFrame(reviews)[
        [
            "vm_id",
            "corrected_label",
        ]
    ]

    dataframe.to_csv(
        FEEDBACK_PATH,
        index=False,
    )


# ============================================================
# Model loading
# ============================================================

_model_cache = {}


def load_saved_model(path):

    saved = joblib.load(path)

    if isinstance(saved, dict):

        if "model" not in saved:

            raise ValueError(
                "Saved model has no model entry."
            )

        return saved["model"]

    return saved


def get_model():

    if "active_model" in _model_cache:

        return _model_cache[
            "active_model"
        ]

    if FEEDBACK_MODEL_PATH.exists():

        model = load_saved_model(
            FEEDBACK_MODEL_PATH
        )

    else:

        if not MODEL_PATH.exists():

            raise FileNotFoundError(
                "Model not found. Run train_model.py first."
            )

        model = load_saved_model(
            MODEL_PATH
        )

    _model_cache[
        "active_model"
    ] = model

    return model


def clear_model_cache():

    _model_cache.pop(
        "active_model",
        None,
    )


def get_applied_feedback_labels():

    if not FEEDBACK_MODEL_PATH.exists():
        return {}

    saved = joblib.load(
        FEEDBACK_MODEL_PATH
    )

    if not isinstance(saved, dict):
        return {}

    stored = saved.get(
        "feedback_labels",
        {},
    )

    return {
        str(vm_id): as_bool(label)
        for vm_id, label
        in stored.items()
    }


def get_pending_reviews():

    reviews = get_reviews()

    applied = (
        get_applied_feedback_labels()
    )

    pending = []

    for review in reviews:

        vm_id = str(
            review["vm_id"]
        )

        corrected_label = as_bool(
            review["corrected_label"]
        )

        if (
            vm_id not in applied
            or applied[vm_id] != corrected_label
        ):

            pending.append(
                review
            )

    return pending


# ============================================================
# Confidence
# ============================================================

def confidence_band(score):

    if score >= HIGH_CONFIDENCE:
        return "High"

    if score >= MEDIUM_CONFIDENCE:
        return "Medium"

    return "Low"


def recommendation_for(
    prediction,
    band,
):

    if not prediction:
        return "No Action"

    if band == "High":
        return "Shutdown"

    if band == "Medium":
        return "Resize"

    return "No Action"


# ============================================================
# Azure-inspired threshold baseline
# ============================================================

def baseline_prediction(row):

    return bool(
        row["cpu_p95"] < 3
        and row[
            "cpu_last_3_days_mean"
        ] <= 2
        and row[
            "outbound_network_mean"
        ] < 2
    )


# ============================================================
# Analyse uploaded/new VM data
# ============================================================

def analyse(dataframe):

    missing = [
        feature
        for feature in FEATURES
        if feature not in dataframe.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    model = get_model()

    X = dataframe[
        FEATURES
    ]

    probabilities = (
        model.predict_proba(X)
    )

    predictions = (
        model.predict(X)
    )

    true_index = list(
        model.classes_
    ).index(True)

    probability_true = probabilities[
        :,
        true_index,
    ]

    confidence_scores = (
        probabilities.max(axis=1)
    )

    bands = [
        confidence_band(score)
        for score
        in confidence_scores
    ]

    recommendations = [
        recommendation_for(
            bool(prediction),
            band,
        )
        for prediction, band
        in zip(
            predictions,
            bands,
        )
    ]

    baseline = dataframe.apply(
        baseline_prediction,
        axis=1,
    )

    output = dataframe.copy()

    output[
        "baseline_underutilised"
    ] = baseline

    output[
        "predicted_underutilised"
    ] = predictions.astype(bool)

    output[
        "probability_underutilised"
    ] = probability_true

    output[
        "confidence"
    ] = confidence_scores

    output[
        "confidence_band"
    ] = bands

    output[
        "recommendation"
    ] = recommendations

    output[
        "models_agree"
    ] = (
        output[
            "baseline_underutilised"
        ]
        ==
        output[
            "predicted_underutilised"
        ]
    )

    return output


# ============================================================
# Research dataset
# ============================================================

_dataset_cache = None


def read_research_dataset():

    dataframe = pd.read_csv(
        DATASET_PATH
    )

    missing = [
        feature
        for feature in FEATURES
        if feature not in dataframe.columns
    ]

    if missing:

        raise ValueError(
            "Dataset missing columns: "
            + ", ".join(missing)
        )

    if "reference_label" not in dataframe.columns:

        raise ValueError(
            "reference_label column missing."
        )

    if "vm_id" not in dataframe.columns:

        dataframe.insert(
            0,
            "vm_id",
            [
                f"vm-{number:04d}"
                for number
                in range(
                    len(dataframe)
                )
            ],
        )

    dataframe["vm_id"] = (
        dataframe["vm_id"]
        .astype(str)
    )

    return dataframe


def labels_with_applied_feedback(
    dataframe,
):

    labels = (
        dataframe[
            "reference_label"
        ]
        .astype(bool)
        .copy()
    )

    feedback = (
        get_applied_feedback_labels()
    )

    vm_ids = (
        dataframe["vm_id"]
        .astype(str)
    )

    for vm_id, corrected_label in feedback.items():

        matches = (
            vm_ids == str(vm_id)
        )

        labels.loc[
            matches
        ] = corrected_label

    return labels


# ============================================================
# Out-of-fold predictions
# ============================================================

def build_oof_dataset(
    dataframe,
    labels,
):

    X = dataframe[
        FEATURES
    ]

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    )

    folds = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    probabilities = cross_val_predict(
        model,
        X,
        labels,
        cv=folds,
        method="predict_proba",
        n_jobs=-1,
    )

    probability_true = (
        probabilities[:, 1]
    )

    predictions = (
        probability_true >= 0.5
    )

    confidence_scores = (
        probabilities.max(axis=1)
    )

    bands = [
        confidence_band(score)
        for score
        in confidence_scores
    ]

    recommendations = [
        recommendation_for(
            bool(prediction),
            band,
        )
        for prediction, band
        in zip(
            predictions,
            bands,
        )
    ]

    baseline = dataframe.apply(
        baseline_prediction,
        axis=1,
    )

    output = dataframe.copy()

    output[
        "active_reference_label"
    ] = labels.astype(bool)

    output[
        "baseline_underutilised"
    ] = baseline

    output[
        "predicted_underutilised"
    ] = predictions.astype(bool)

    output[
        "probability_underutilised"
    ] = probability_true

    output[
        "confidence"
    ] = confidence_scores

    output[
        "confidence_band"
    ] = bands

    output[
        "recommendation"
    ] = recommendations

    output[
        "models_agree"
    ] = (
        output[
            "baseline_underutilised"
        ]
        ==
        output[
            "predicted_underutilised"
        ]
    )

    output[
        "_vm_sort"
    ] = numeric_vm_id(
        output["vm_id"]
    )

    return output


def load_dataset():

    global _dataset_cache

    if _dataset_cache is not None:

        return (
            _dataset_cache.copy()
        )

    dataframe = (
        read_research_dataset()
    )

    labels = (
        labels_with_applied_feedback(
            dataframe
        )
    )

    output = build_oof_dataset(
        dataframe,
        labels,
    )

    _dataset_cache = (
        output.copy()
    )

    return output


# ============================================================
# Feedback model
# ============================================================

def train_feedback_model(
    dataframe,
    corrected_labels,
    feedback_labels,
):

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(
        dataframe[FEATURES],
        corrected_labels,
    )

    MODEL_DIR.mkdir(
        exist_ok=True
    )

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "feedback_labels":
                feedback_labels,
            "review_count":
                len(feedback_labels),
            "training_rows":
                len(dataframe),
        },
        FEEDBACK_MODEL_PATH,
    )


# ============================================================
# Retrain
# ============================================================

@app.route(
    "/retrain",
    methods=["POST"],
)
def retrain():

    global _dataset_cache

    pending_reviews = (
        get_pending_reviews()
    )

    if not pending_reviews:

        flash(
            "There are no new reviews to retrain with."
        )

        return redirect(
            url_for("review")
        )

    before_dataset = (
        load_dataset()
    )

    before_confidence = {
        str(row["vm_id"]):
        float(row["confidence"])
        for _, row
        in before_dataset.iterrows()
    }

    dataframe = (
        read_research_dataset()
    )

    corrected_labels = (
        labels_with_applied_feedback(
            dataframe
        )
    )

    feedback_labels = (
        get_applied_feedback_labels()
    )

    vm_ids = (
        dataframe["vm_id"]
        .astype(str)
    )

    for saved_review in pending_reviews:

        vm_id = str(
            saved_review["vm_id"]
        )

        corrected_label = as_bool(
            saved_review[
                "corrected_label"
            ]
        )

        matches = (
            vm_ids == vm_id
        )

        if matches.any():

            corrected_labels.loc[
                matches
            ] = corrected_label

            feedback_labels[
                vm_id
            ] = corrected_label

    if not feedback_labels:

        flash(
            "No reviews matched the dataset."
        )

        return redirect(
            url_for("review")
        )

    # Save a separate feedback-adapted model.
    train_feedback_model(
        dataframe,
        corrected_labels,
        feedback_labels,
    )

    # Clear the old model from memory.
    clear_model_cache()

    # Clear cached OOF predictions.
    _dataset_cache = None

    # Recalculate predictions after feedback.
    after_dataset = (
        load_dataset()
    )

    after_confidence = {
        str(row["vm_id"]):
        float(row["confidence"])
        for _, row
        in after_dataset.iterrows()
    }

    comparisons = []

    for saved_review in pending_reviews:

        vm_id = str(
            saved_review["vm_id"]
        )

        if (
            vm_id in before_confidence
            and
            vm_id in after_confidence
        ):

            comparisons.append(
                {
                    "vm_id":
                        vm_id,

                    "before":
                        before_confidence[
                            vm_id
                        ],

                    "after":
                        after_confidence[
                            vm_id
                        ],
                }
            )

    comparisons.sort(
        key=lambda item:
        item["before"]
    )

    session[
        "retrain_summary"
    ] = {
        "comparisons":
            comparisons[:5],
        "review_count":
            len(pending_reviews),
    }

    flash(
        "Feedback retraining completed."
    )

    return redirect(
        url_for("review")
    )


# ============================================================
# Dashboard
# ============================================================

@app.route("/")
def dashboard():

    dataframe = (
        load_dataset()
    )

    selected_recommendation = (
        request.args.get(
            "recommendation",
            "",
        )
    )

    selected_confidence = (
        request.args.get(
            "confidence",
            "",
        )
    )

    selected_comparison = (
        request.args.get(
            "comparison",
            "",
        )
    )

    view = dataframe.copy()

    if selected_recommendation:

        view = view[
            view["recommendation"]
            == selected_recommendation
        ]

    if selected_confidence:

        view = view[
            view["confidence_band"]
            == selected_confidence
        ]

    if selected_comparison == "disagree":

        view = view[
            ~view["models_agree"]
        ]

    statistics = {
        "total":
            len(dataframe),

        "shutdown":
            int(
                (
                    dataframe[
                        "recommendation"
                    ]
                    == "Shutdown"
                ).sum()
            ),

        "resize":
            int(
                (
                    dataframe[
                        "recommendation"
                    ]
                    == "Resize"
                ).sum()
            ),

        "model_disagreements":
            int(
                (
                    ~dataframe[
                        "models_agree"
                    ]
                ).sum()
            ),
    }

    # Lowest-confidence VMs first.
    # Agreements and disagreements remain mixed.
    view = view.sort_values(
        "confidence",
        ascending=True,
    )

    total_matching = (
        len(view)
    )

    rows = (
        view
        .head(100)
        .to_dict("records")
    )

    return render_template(
        "dashboard.html",

        rows=rows,
        stats=statistics,

        showing=len(rows),
        total_matching=total_matching,

        recommendation=(
            selected_recommendation
        ),

        confidence=(
            selected_confidence
        ),

        comparison=(
            selected_comparison
        ),
    )


# ============================================================
# Sample CSV
# ============================================================

@app.route("/sample")
def sample_csv():

    dataframe = (
        load_dataset()
    )

    selected = []
    used_vm_ids = set()

    def add_vm(rows):

        if rows.empty:
            return

        row = rows.iloc[0]

        vm_id = str(
            row["vm_id"]
        )

        if vm_id not in used_vm_ids:

            selected.append(
                row
            )

            used_vm_ids.add(
                vm_id
            )

    # 1. Clear Shutdown example
    add_vm(
        dataframe[
            dataframe[
                "recommendation"
            ] == "Shutdown"
        ].sort_values(
            "confidence",
            ascending=False,
        )
    )

    # 2. Resize example
    add_vm(
        dataframe[
            dataframe[
                "recommendation"
            ] == "Resize"
        ].sort_values(
            "confidence",
            ascending=False,
        )
    )

    # 3. Clear No Action example
    add_vm(
        dataframe[
            dataframe[
                "recommendation"
            ] == "No Action"
        ].sort_values(
            "confidence",
            ascending=False,
        )
    )

    # 4. Baseline / Random Forest disagreement
    add_vm(
        dataframe[
            ~dataframe[
                "models_agree"
            ]
        ].sort_values(
            "confidence",
            ascending=True,
        )
    )

    # 5. Low-confidence example
    add_vm(
        dataframe.sort_values(
            "confidence",
            ascending=True,
        )
    )

    # Fill remaining spaces if two categories
    # selected the same VM.
    if len(selected) < 5:

        remaining = dataframe[
            ~dataframe[
                "vm_id"
            ]
            .astype(str)
            .isin(
                used_vm_ids
            )
        ].sort_values(
            "confidence",
            ascending=True,
        )

        for _, row in remaining.iterrows():

            selected.append(
                row
            )

            used_vm_ids.add(
                str(row["vm_id"])
            )

            if len(selected) == 5:
                break

    sample = pd.DataFrame(
        selected
    )

    # Only export fields needed by the upload page.
    # Do not expose reference labels or model outputs.
    export_columns = [
        "vm_id",
        *FEATURES,
    ]

    sample = sample[
        export_columns
    ]

    return Response(
        sample.to_csv(
            index=False
        ),
        mimetype="text/csv",
        headers={
            "Content-Disposition":
                "attachment; filename=sample_vms.csv"
        },
    )


# ============================================================
# Upload
# ============================================================

@app.route(
    "/upload",
    methods=["GET", "POST"],
)
def upload():

    results = None
    error = None

    if request.method == "POST":

        uploaded_file = (
            request.files.get(
                "file"
            )
        )

        if (
            not uploaded_file
            or
            not uploaded_file.filename
        ):

            error = (
                "Choose a CSV file first."
            )

        else:

            try:

                dataframe = (
                    pd.read_csv(
                        uploaded_file
                    )
                )

                if "vm_id" not in dataframe.columns:

                    dataframe.insert(
                        0,
                        "vm_id",
                        [
                            f"uploaded-{index:04d}"
                            for index
                            in range(
                                len(dataframe)
                            )
                        ],
                    )

                results = (
                    analyse(
                        dataframe
                    )
                    .to_dict(
                        "records"
                    )
                )

            except ValueError as exception:

                error = str(
                    exception
                )

            except Exception as exception:

                error = (
                    "The CSV could not be analysed. "
                    + str(exception)
                )

    return render_template(
        "upload.html",

        results=results,
        error=error,
        features=FEATURES,
    )


# ============================================================
# Human review
# ============================================================

@app.route(
    "/review",
    methods=["GET", "POST"],
)
def review():

    if request.method == "POST":

        save_review(
            vm_id=request.form[
                "vm_id"
            ],

            prediction=request.form[
                "prediction"
            ],

            recommendation=request.form[
                "recommendation"
            ],

            confidence=request.form[
                "confidence"
            ],

            corrected_label=request.form[
                "corrected_label"
            ],

            reviewer_notes=(
                request.form.get(
                    "reviewer_notes",
                    "",
                )
            ),
        )

        flash(
            "Feedback saved for "
            + request.form[
                "vm_id"
            ]
            + "."
        )

        return redirect(
            url_for("review")
        )

    dataframe = (
        load_dataset()
    )

    reviews = (
        get_reviews()
    )

    pending_reviews = (
        get_pending_reviews()
    )

    reviewed_vm_ids = {
        str(
            saved_review[
                "vm_id"
            ]
        )
        for saved_review
        in reviews
    }

    review_view = dataframe[
        ~dataframe[
            "vm_id"
        ]
        .astype(str)
        .isin(
            reviewed_vm_ids
        )
    ].copy()

    review_view = (
        review_view.sort_values(
            "confidence",
            ascending=True,
        )
    )

    review_queue = (
        review_view
        .head(20)
        .to_dict(
            "records"
        )
    )

    retrain_summary = (
        session.pop(
            "retrain_summary",
            None,
        )
    )

    return render_template(
        "review.html",

        queue=review_queue,

        reviews=reviews,

        pending_reviews=(
            pending_reviews
        ),

        retrain_summary=(
            retrain_summary
        ),

        feedback_model_active=(
            FEEDBACK_MODEL_PATH.exists()
        ),
    )


# ============================================================
# Health check
# ============================================================

@app.route("/health")
def health():

    return {
        "status":
            "healthy",

        "frozen_model_available":
            MODEL_PATH.exists(),

        "feedback_model_active":
            FEEDBACK_MODEL_PATH.exists(),

        "application":
            "Beyond Thresholds",
    }


# ============================================================
# Start
# ============================================================

init_database()


if __name__ == "__main__":

    app.run(
        debug=True
    )