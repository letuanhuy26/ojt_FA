# SPY RC-SFG reproduction source

Code and configuration for the SPY regime-conditioned feature-gating (RC-SFG) pilot. This repository intentionally contains source, experiment configs, dependency files, and the small frozen-candidate metadata needed by the held-out runner. It does not contain price data, prediction/result tables, figures, or the paper.

## Setup

Python 3.12 is the environment used for the original run.

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

Optional XGBoost comparison:

~~~powershell
pip install -r requirements-exploratory.txt
~~~

## Data and rights

The downloader retrieves SPY daily bars from Yahoo Finance and writes a local audit manifest:

~~~powershell
python scripts/download_data.py
python scripts/prepare_features.py
~~~

The source project's data rights and redistribution status are unresolved. No price data are included here. A new Yahoo download may differ from the historical snapshot because the vendor can revise data. The frozen confirmation/final runner checks SHA-256 hashes for the original raw and feature files; exact held-out reproduction requires those exact files from an authorized source. The freeze metadata records the expected hashes. Do not rerun confirmation/final on revised data and treat it as the original one-look evaluation.

## Main development run

~~~powershell
python scripts/run_baselines.py
python scripts/run_gate_ablations.py
python scripts/run_gate_diagnostics.py
python scripts/run_hard_state_development.py
python scripts/run_stability_gate_comparison.py
~~~

The scripts write generated files under results/ and cached data under data/processed/; these are ignored by Git. The broader post-hoc scripts also require their documented upstream outputs and, for some checks, the frozen metadata.

## Held-out evaluation

Only with the original authorized data snapshot, run each split once and in order:

~~~powershell
python scripts/run_heldout.py --split confirmation
python scripts/run_heldout.py --split final
python scripts/analyze_heldout.py
python scripts/check_protocol.py
~~~

The held-out code checks the frozen input hashes and refuses repeated or out-of-order evaluations. The public code alone cannot recreate the historical numbers without the matching data snapshot.

## Reproducibility limits

- The historical report used SPY from 1993–2025, a fixed chronological walk-forward protocol, and a frozen RC-SFG configuration.
- Yahoo data and manifests are not bundled as market-data files; manifests are audit metadata only.
- Runtime, hardware, and vendor corrections can affect results. Keep the seed and protocol fixed when comparing runs.
