"""Train and compare models, write figures and save the best estimator.

Run from the project root:

    python -m src.train
"""

from __future__ import annotations

import argparse

import joblib
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split

from . import config, data, evaluate, features

# Both estimators handle NaN natively, so rows with missing coordinates or
# ratings stay in the training set instead of being dropped or imputed.
# (The classic GradientBoostingRegressor does not, which is why the histogram
# variant is used here.)
MODELS = {
    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
    ),
    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.1,
        random_state=config.RANDOM_STATE,
    ),
}


def main(csv_path=None) -> None:
    print("Loading and cleaning data...")
    df = data.load_clean(csv_path)
    X, y = features.build(df)
    print(f"  {len(X):,} rows, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    scores = {}
    fitted = {}
    for name, model in MODELS.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        scores[name] = evaluate.score(y_test, model.predict(X_test))
        fitted[name] = model

    print("\nTest set results\n")
    print(evaluate.results_table(scores))

    best_name = min(scores, key=lambda name: scores[name]["MAE"])
    best_model = fitted[best_name]
    print(f"\nBest model by MAE: {best_name}")

    y_pred = best_model.predict(X_test)
    evaluate.plot_predicted_vs_actual(y_test, y_pred, best_name)
    evaluate.plot_residuals(y_test, y_pred, best_name)
    top_features = evaluate.plot_feature_importance(best_model, X_test, y_test, best_name)
    print(f"Figures written to {config.FIGURES_DIR}")

    print("\nTop features")
    print(top_features.round(4).to_string())

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODELS_DIR / "model.joblib"
    joblib.dump({"model": best_model, "features": list(X.columns)}, model_path)
    print(f"\nSaved {best_name} to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train food delivery time models.")
    parser.add_argument(
        "--data",
        default=None,
        help=f"Path to the training CSV (default: {config.RAW_DATA}).",
    )
    args = parser.parse_args()
    main(args.data)
