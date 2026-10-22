from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data.features import FEATURE_COLUMNS, REGIME_COLUMNS
from src.models.ours import FeatureGateLSTM, loss_with_gate_penalties
from src.evaluation.regime_cache import fold_regime_posteriors


ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_examples(path: Path | None = None) -> pd.DataFrame:
    path = path or ROOT / "data/processed/SPY_features_1993-2025.csv"
    frame = pd.read_csv(path, parse_dates=["date", "target_date"])
    required = set(FEATURE_COLUMNS + REGIME_COLUMNS + ["date", "target_date", "target", "target_return"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Feature table missing columns: {missing}")
    return frame


def chronological_folds(protocol: dict, split: str = "development") -> list[tuple[int, pd.Timestamp, pd.Timestamp]]:
    evaluation = protocol["evaluation"]
    if split == "development":
        return [
            (year, pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year + 1}-01-01"))
            for year in evaluation["development_years"]
        ]
    if split == "confirmation":
        start = pd.Timestamp(evaluation["confirmation_start"])
        end = pd.Timestamp(evaluation["confirmation_end_exclusive"])
        return [(int(start.year), start, end)]
    if split == "final":
        start = pd.Timestamp(evaluation["final_start"])
        end = pd.Timestamp(evaluation["final_end_exclusive"])
        return [(int(start.year), start, end)]
    raise ValueError(f"Unknown evaluation split: {split}")


def prepare_fold(frame: pd.DataFrame, protocol: dict, year: int, start: pd.Timestamp,
                 end: pd.Timestamp, window: int) -> dict:
    target_dates = pd.to_datetime(frame["target_date"])
    train_mask = (target_dates >= pd.Timestamp(protocol["evaluation"]["train_start"])) & (target_dates < start)
    # Only the training period and this validation period enter the filter.
    all_mask = (target_dates >= pd.Timestamp(protocol["evaluation"]["train_start"])) & (target_dates < end)
    val_mask = (target_dates >= start) & (target_dates < end)
    if int(train_mask.sum()) < 1000 or int(val_mask.sum()) < 100:
        raise ValueError(f"Insufficient rows for {year}: train={train_mask.sum()} val={val_mask.sum()}")
    all_positions = np.flatnonzero(all_mask.to_numpy())
    train_in_all = train_mask.to_numpy()[all_positions]
    val_in_all = val_mask.to_numpy()[all_positions]
    if not np.all(np.diff(all_positions) == 1):
        raise ValueError("Feature rows in fold are not contiguous; inspect the source calendar and missing-data policy.")

    x_raw = frame.loc[all_mask, FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    z_raw = frame.loc[all_mask, REGIME_COLUMNS].to_numpy(dtype=np.float64)
    imputer_x = SimpleImputer(strategy="median").fit(x_raw[train_in_all])
    x_raw = imputer_x.transform(x_raw)
    x_scaler = StandardScaler().fit(x_raw[train_in_all])
    x_scaled = x_scaler.transform(x_raw).astype(np.float32)

    posterior, regime_details = fold_regime_posteriors(
        year, start, end, z_raw, train_in_all, protocol,
    )

    # A sequence ending at row i predicts the label attached to row i.
    sequence_x, sequence_p, sequence_y, sequence_meta = [], [], [], []
    frame_fold = frame.iloc[all_positions].reset_index(drop=True)
    y_all = frame_fold["target"].to_numpy(dtype=np.float32)
    for i in range(window - 1, len(frame_fold)):
        sequence_x.append(x_scaled[i - window + 1 : i + 1])
        sequence_p.append(posterior[i - window + 1 : i + 1].astype(np.float32))
        sequence_y.append(y_all[i])
        sequence_meta.append(i)
    seq_positions = np.asarray(sequence_meta, dtype=int)
    sample_train = train_in_all[seq_positions]
    sample_val = val_in_all[seq_positions]
    y_seq = np.asarray(sequence_y, dtype=np.float32)
    val_row_positions = np.flatnonzero(val_in_all)
    val_target_dates = frame_fold.loc[val_row_positions, "target_date"].to_numpy()
    return {
        "year": year,
        "x_train_rows": x_scaled[train_in_all],
        "x_val_rows": x_scaled[val_in_all],
        "y_train_rows": frame_fold.loc[train_in_all, "target"].to_numpy(dtype=np.int8),
        "y_val_rows": frame_fold.loc[val_in_all, "target"].to_numpy(dtype=np.int8),
        "previous_direction_val": (frame_fold.loc[val_in_all, "intraday_return"].to_numpy() > 0).astype(np.int8),
        "x_sequences_train": np.asarray(sequence_x, dtype=np.float32)[sample_train],
        "p_sequences_train": np.asarray(sequence_p, dtype=np.float32)[sample_train],
        "y_sequences_train": y_seq[sample_train],
        "x_sequences_val": np.asarray(sequence_x, dtype=np.float32)[sample_val],
        "p_sequences_val": np.asarray(sequence_p, dtype=np.float32)[sample_val],
        "y_sequences_val": y_seq[sample_val].astype(np.int8),
        "val_target_dates": pd.to_datetime(val_target_dates),
        "hmm_train_log_likelihood_per_obs": regime_details["log_likelihood_per_obs"],
        "hmm_iterations": regime_details["iterations"],
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "x_mean": x_scaler.mean_.tolist(),
        "x_scale": x_scaler.scale_.tolist(),
    }


def metric_row(y_true: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=np.int8)
    probability = np.asarray(probability, dtype=np.float64)
    predicted = (probability >= threshold).astype(np.int8)
    auc = roc_auc_score(y_true, probability) if len(np.unique(y_true)) == 2 else float("nan")
    return {
        "mcc": float(matthews_corrcoef(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "auroc": float(auc),
        "brier": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, np.clip(probability, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "positive_rate": float(y_true.mean()),
        "predicted_positive_rate": float(predicted.mean()),
        "n": int(len(y_true)),
    }


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _device(config: dict) -> torch.device:
    choice = config.get("device", "auto")
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def fit_neural(fold: dict, config: dict, mode: str) -> tuple[np.ndarray, dict]:
    seed = int(config["seed"]) + int(fold["year"])
    _seed_everything(seed)
    device = _device(config)
    x_train = torch.from_numpy(fold["x_sequences_train"]).to(device)
    p_train = torch.from_numpy(fold["p_sequences_train"]).to(device)
    y_train = torch.from_numpy(fold["y_sequences_train"]).to(device)
    dataset = TensorDataset(x_train, p_train, y_train)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=True, generator=generator)
    model = FeatureGateLSTM(
        n_features=x_train.shape[-1],
        n_regimes=p_train.shape[-1],
        hidden_size=int(config["hidden_size"]),
        gate_hidden_size=int(config["gate_hidden_size"]),
        dropout=float(config["dropout"]),
        mode=mode,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    start = time.perf_counter()
    for _epoch in range(int(config["epochs"])):
        model.train()
        for xb, pb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits, gates = model(xb, pb)
            penalty_s = float(config["lambda_sparsity"]) if mode in {"static_gate_lstm", "dynamic_gate_lstm", "regime_dynamic_gate_lstm"} else 0.0
            penalty_t = float(config["lambda_stability"]) if mode in {"dynamic_gate_lstm", "regime_dynamic_gate_lstm"} else 0.0
            loss = loss_with_gate_penalties(logits, yb, gates, penalty_s, penalty_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    model.eval()
    with torch.no_grad():
        xv = torch.from_numpy(fold["x_sequences_val"]).to(device)
        pv = torch.from_numpy(fold["p_sequences_val"]).to(device)
        logits, gates = model(xv, pv)
        probability = torch.sigmoid(logits).cpu().numpy()
        gate_average = gates.mean(dim=(0, 1)).cpu().numpy()
        gate_temporal_change = (gates[:, 1:] - gates[:, :-1]).abs().mean().item() if gates.shape[1] > 1 else 0.0
    return probability, {
        "device": str(device),
        "runtime_seconds": time.perf_counter() - start,
        "gate_mean": float(gate_average.mean()),
        "effective_features_ge_0_5": int((gate_average >= 0.5).sum()),
        "gate_temporal_change": float(gate_temporal_change),
        "seed": seed,
        "epochs": int(config["epochs"]),
    }


def fit_non_neural(fold: dict, model_name: str, seed: int) -> tuple[np.ndarray, dict]:
    start = time.perf_counter()
    if model_name == "majority":
        probability = np.full(len(fold["y_val_rows"]), float(fold["y_train_rows"].mean()))
        metadata = {"device": "cpu", "seed": seed}
    elif model_name == "previous_direction":
        probability = fold["previous_direction_val"].astype(np.float64)
        metadata = {"device": "cpu", "seed": seed}
    elif model_name == "logistic":
        model = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
        model.fit(fold["x_train_rows"], fold["y_train_rows"])
        probability = model.predict_proba(fold["x_val_rows"])[:, 1]
        metadata = {"device": "cpu", "seed": seed}
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=20,
            n_jobs=-1, random_state=seed,
        )
        model.fit(fold["x_train_rows"], fold["y_train_rows"])
        probability = model.predict_proba(fold["x_val_rows"])[:, 1]
        metadata = {"device": "cpu", "seed": seed}
    else:
        raise ValueError(f"Not a non-neural baseline: {model_name}")
    metadata["runtime_seconds"] = time.perf_counter() - start
    return probability, metadata


def evaluate_models(models: list[str], config: dict | None = None,
                    protocol: dict | None = None, data_path: Path | None = None,
                    split: str = "development") -> tuple[pd.DataFrame, pd.DataFrame]:
    protocol = protocol or load_yaml(ROOT / "configs/protocol.yaml")
    config = config or load_yaml(ROOT / "configs/ours.yaml")
    frame = load_examples(data_path)
    window = int(config.get("window", 20))
    rows, prediction_rows = [], []
    base_seed = int(protocol["evaluation"]["random_seed"])
    for year, start, end in chronological_folds(protocol, split):
        fold = prepare_fold(frame, protocol, year, start, end, window)
        for model_name in models:
            if model_name in FeatureGateLSTM.MODES:
                probability, metadata = fit_neural(fold, config, model_name)
                y_true = fold["y_sequences_val"]
                dates = fold["val_target_dates"]
            else:
                probability, metadata = fit_non_neural(fold, model_name, base_seed + year)
                y_true = fold["y_val_rows"]
                dates = fold["val_target_dates"]
            scores = metric_row(y_true, probability, threshold=float(protocol["evaluation"]["threshold"]))
            rows.append({
                "model": model_name, "fold_year": year,
                "hmm_iterations": fold["hmm_iterations"],
                "hmm_loglik_per_obs": fold["hmm_train_log_likelihood_per_obs"],
                **scores, **metadata,
            })
            prediction_rows.extend({
                "model": model_name,
                "fold_year": year,
                "target_date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "y_true": int(actual),
                "probability": float(prob),
            } for date, actual, prob in zip(dates, y_true, probability))
    return pd.DataFrame(rows), pd.DataFrame(prediction_rows)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.groupby("model", sort=False).agg(
        folds=("fold_year", "count"),
        mean_mcc=("mcc", "mean"),
        sd_fold_mcc=("mcc", "std"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        mean_auroc=("auroc", "mean"),
        mean_brier=("brier", "mean"),
        mean_log_loss=("log_loss", "mean"),
        mean_positive_rate=("positive_rate", "mean"),
        mean_runtime_seconds=("runtime_seconds", "mean"),
    ).reset_index()


def write_run_metadata(path: Path, metrics: pd.DataFrame, models: list[str],
                       split: str = "development") -> None:
    metadata = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "models": models,
        "fold_years": [int(v) for v in sorted(metrics["fold_year"].unique())],
        "metric": "Matthews correlation coefficient, fixed threshold 0.5",
        "data_split": split,
        "device": sorted(str(v) for v in metrics["device"].dropna().unique()) if "device" in metrics else [],
        "rows": int(len(metrics)),
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
