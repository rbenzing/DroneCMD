"""
Classifier training pipeline for DroneCMD protocol classification.

Trains an ensemble of sklearn classifiers on labeled IQ captures, evaluates
with cross-validation, and saves the trained models + metadata to disk.

Usage:
    dronecmd train --data-dir captures/ --output-dir models/
    python -m dronecmd.training.train --data-dir captures/ --output-dir models/

The output directory will contain:
    random_forest.pkl       - Primary RandomForest model
    logistic_regression.pkl - Secondary LR model (ensemble)
    svm.pkl                 - Secondary SVM model (ensemble)
    scaler.pkl              - Feature scaler (StandardScaler)
    label_encoder.pkl       - LabelEncoder
    model_metadata.json     - Training metrics, feature names, class list
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
        import joblib  # noqa: F401
    except ImportError:
        print(
            "ERROR: scikit-learn and joblib are required for training.\n"
            "       pip install -e '.[dev]'  or  pip install scikit-learn joblib",
            file=sys.stderr,
        )
        sys.exit(1)


def train(
    data_dir: str | Path,
    output_dir: str | Path,
    cv_folds: int = 5,
    n_jobs: int = -1,
    random_state: int = 42,
) -> Dict:
    """Train classifiers and save to output_dir.

    Args:
        data_dir: Directory of labeled IQ captures (see docs/iq_capture_guide.md).
        output_dir: Directory to write trained model files.
        cv_folds: Number of cross-validation folds for evaluation.
        n_jobs: Parallel jobs for sklearn (-1 = all CPUs).
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary of training metrics (also written to model_metadata.json).

    Raises:
        FileNotFoundError: If data_dir does not exist.
        ValueError: If not enough data to train.
        ImportError: If scikit-learn or joblib are not installed.
    """
    _require_sklearn()

    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.svm import SVC

    from .dataset import build_dataset

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build dataset
    # ------------------------------------------------------------------
    logger.info(f"Building dataset from {data_dir} ...")
    X, y_str, stats, feature_names = build_dataset(data_dir)

    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Class distribution: {stats.class_counts}")

    if X.shape[0] < cv_folds * 2:
        raise ValueError(
            f"Need at least {cv_folds * 2} samples for {cv_folds}-fold CV, "
            f"got {X.shape[0]}.  Capture more data."
        )

    # ------------------------------------------------------------------
    # Encode labels
    # ------------------------------------------------------------------
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    classes = list(le.classes_)
    logger.info(f"Classes: {classes}")

    # ------------------------------------------------------------------
    # Scale features
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ------------------------------------------------------------------
    # Define models
    # ------------------------------------------------------------------
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=n_jobs,
            random_state=random_state,
            class_weight="balanced",
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            multi_class="multinomial",
            n_jobs=n_jobs,
            random_state=random_state,
            class_weight="balanced",
        ),
        "svm": SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        ),
    }

    # ------------------------------------------------------------------
    # Cross-validation evaluation
    # ------------------------------------------------------------------
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores: Dict[str, Dict] = {}

    for name, model in models.items():
        logger.info(f"Cross-validating {name} ({cv_folds} folds) ...")
        start = time.perf_counter()
        scores = cross_val_score(
            model, X_scaled, y, cv=cv, scoring="f1_weighted", n_jobs=n_jobs
        )
        elapsed = time.perf_counter() - start
        cv_scores[name] = {
            "f1_weighted_mean": float(scores.mean()),
            "f1_weighted_std": float(scores.std()),
            "cv_time_s": round(elapsed, 2),
        }
        logger.info(
            f"  {name}: F1 = {scores.mean():.3f} ± {scores.std():.3f}  ({elapsed:.1f}s)"
        )

    # ------------------------------------------------------------------
    # Train final models on full dataset
    # ------------------------------------------------------------------
    trained_models: Dict = {}
    for name, model in models.items():
        logger.info(f"Training final {name} on full dataset ...")
        model.fit(X_scaled, y)
        trained_models[name] = model

    # ------------------------------------------------------------------
    # Full-dataset classification report (for audit trail)
    # ------------------------------------------------------------------
    rf_preds = trained_models["random_forest"].predict(X_scaled)
    report = classification_report(y, rf_preds, target_names=classes, output_dict=True)

    # ------------------------------------------------------------------
    # Save models and metadata
    # ------------------------------------------------------------------
    for name, model in trained_models.items():
        path = output_dir / f"{name}.pkl"
        joblib.dump(model, path)
        logger.info(f"Saved {path}")

    joblib.dump(scaler, output_dir / "scaler.pkl")
    joblib.dump(le, output_dir / "label_encoder.pkl")
    logger.info(f"Saved scaler and label encoder to {output_dir}")

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir.resolve()),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_names": feature_names,
        "classes": classes,
        "class_counts": {k: int(v) for k, v in stats.class_counts.items()},
        "cv_folds": cv_folds,
        "cv_scores": cv_scores,
        "random_forest_classification_report": report,
        "dataset_stats": {
            "total_files": stats.total_files,
            "failed_files": stats.failed_files,
            "total_packets": stats.total_packets,
        },
    }

    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Saved training metadata to {metadata_path}")

    # Print summary
    best_name = max(cv_scores, key=lambda k: cv_scores[k]["f1_weighted_mean"])
    best_f1 = cv_scores[best_name]["f1_weighted_mean"]
    print(f"\n{'='*60}")
    print(f"Training complete.  Models saved to: {output_dir}")
    print(f"Best model: {best_name}  (F1 = {best_f1:.3f})")
    print(f"Classes: {classes}")
    print(f"{'='*60}\n")

    return metadata


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Train DroneCMD protocol classifiers from labeled IQ captures."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing labeled IQ captures (see docs/iq_capture_guide.md)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write trained model files",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Parallel jobs (-1 = all CPUs, default: -1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    try:
        train(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            cv_folds=args.cv_folds,
            n_jobs=args.jobs,
            random_state=args.seed,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
