from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.regimes.hmm import DiagonalGaussianHMM


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fold_regime_posteriors(year: int, start: pd.Timestamp, end: pd.Timestamp,
                           z_raw: np.ndarray, train_mask: np.ndarray,
                           protocol: dict) -> tuple[np.ndarray, dict]:
    """Return cached forward-filtered HMM probabilities or fit them once."""
    feature_manifest_path = ROOT / "data/manifests/SPY_features_manifest.json"
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    h = protocol["regime"]
    seed = int(protocol["evaluation"]["random_seed"])
    fingerprint_record = {
        "feature_sha256": feature_manifest["feature_sha256"],
        "regime_config": h,
        "seed": seed,
        "fold_year": int(year),
        "train_start": str(protocol["evaluation"]["train_start"]),
        "filter_start": start.strftime("%Y-%m-%d"),
        "filter_end_exclusive": end.strftime("%Y-%m-%d"),
        "training_rows": int(train_mask.sum()),
        "filter_rows": int(len(z_raw)),
    }
    fingerprint = hashlib.sha256(json.dumps(fingerprint_record, sort_keys=True).encode()).hexdigest()
    cache_path = ROOT / f"data/processed/regime_posterior_{year}.npz"
    manifest_path = ROOT / "data/manifests/regime_cache_manifest.json"
    development_fold = int(year) in protocol["evaluation"]["development_years"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if development_fold and manifest_path.exists() else {"folds": {}}
    record = manifest.get("folds", {}).get(str(year), {})

    if development_fold and cache_path.exists() and record.get("fingerprint") == fingerprint and record.get("sha256") == sha256(cache_path):
        with np.load(cache_path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata_json"].item()))
            posterior = saved["posterior"].copy()
        if metadata.get("fingerprint") == fingerprint and posterior.shape == (len(z_raw), int(h["states"])):
            if np.isfinite(posterior).all() and np.allclose(posterior.sum(axis=1), 1.0, atol=1e-5):
                return posterior, metadata["details"]

    scaler = StandardScaler().fit(z_raw[train_mask])
    z_scaled = scaler.transform(z_raw)
    hmm = DiagonalGaussianHMM(
        n_states=int(h["states"]),
        max_iter=int(h["em_iterations"]),
        variance_floor=float(h["covariance_floor"]),
        random_state=seed,
    ).fit(z_scaled[train_mask])
    posterior = hmm.filter(z_scaled)
    details = {
        "log_likelihood_per_obs": float(hmm.log_likelihood_ / int(train_mask.sum())),
        "iterations": int(hmm.n_iter_),
        "training_rows": int(train_mask.sum()),
        "filter_rows": int(len(z_raw)),
    }
    metadata = {
        "fingerprint": fingerprint,
        "fingerprint_record": fingerprint_record,
        "details": details,
        "first_filter_feature_date": None,
        "last_filter_target_year": int(year),
    }
    if development_fold:
        tmp_path = cache_path.with_name(cache_path.stem + ".tmp.npz")
        np.savez_compressed(tmp_path, posterior=posterior, metadata_json=json.dumps(metadata, sort_keys=True))
        os.replace(tmp_path, cache_path)
        manifest.setdefault("folds", {})[str(year)] = {
            "file": str(cache_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(cache_path),
            "fingerprint": fingerprint,
            "fold_year": int(year),
            "filter_end_exclusive": end.strftime("%Y-%m-%d"),
            "last_filter_target_year": int(year),
            **details,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return posterior, details
