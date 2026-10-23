"""Selectors never receive test labels, true coverage, or scenario identity."""

from __future__ import annotations

import numpy as np

TOLERANCE = 1e-12


def _maximum(values: np.ndarray, eligible: np.ndarray | None = None) -> int:
    indices = np.arange(len(values)) if eligible is None else np.flatnonzero(eligible)
    if not len(indices):
        raise ValueError("Empty candidate set")
    best = np.max(values[indices])
    # Candidate arrays must be sorted by canonical ID before entering selectors.
    return int(indices[np.flatnonzero(values[indices] >= best - TOLERANCE)[0]])


def choose(method: str, external: np.ndarray, internal_raw: np.ndarray,
           internal_scaled: np.ndarray, bootstrap: np.ndarray | None,
           *, fixed_window: float = 0.05, adaptive_multiplier: float = 1.0,
           tuned_lambda: float | None = None) -> dict:
    external = np.asarray(external, dtype=float)
    if not all(np.all(np.isfinite(a)) for a in [external, internal_raw, internal_scaled]):
        raise ValueError("Nonfinite selector input")
    # A single externally best candidate is fixed before computing M6's
    # bootstrap differences; the leader is not reselected in each draw.
    leader = _maximum(external)
    delta = external[leader] - external
    eligible = np.ones(len(external), dtype=bool)
    sigma = None
    if method == "M0":
        selected = _maximum(internal_raw)
    elif method == "M1":
        selected = leader
    elif method == "M2":
        eligible = delta <= TOLERANCE
        selected = _maximum(internal_raw, eligible)
    elif method in {"M3", "M4"}:
        weight = 0.5 if method == "M3" else tuned_lambda
        if weight is None or not 0 <= weight <= 1:
            raise ValueError("M4 requires a previously frozen development lambda")
        score = (1 - weight) * internal_scaled + weight * (external + 1) / 2
        selected = _maximum(score)
    elif method == "M5":
        eligible = delta <= fixed_window + TOLERANCE
        selected = _maximum(internal_raw, eligible)
    elif method == "M6":
        if bootstrap is None or len(bootstrap) < 2:
            raise ValueError("M6 requires at least two paired bootstrap draws")
        # Paired differences share the same bootstrap draw for both candidates.
        # Their SD can be much smaller than two independent marginal SDs.
        sigma = (bootstrap[:, leader, None] - bootstrap).std(axis=0, ddof=1)
        eligible = delta <= adaptive_multiplier * sigma + TOLERANCE
        selected = _maximum(internal_raw, eligible)
    elif method == "M7":
        if bootstrap is None or not len(bootstrap):
            raise ValueError("M7 requires bootstrap draws")
        winners = bootstrap >= bootstrap.max(axis=1, keepdims=True) - TOLERANCE
        # Split a draw's vote among exact ties instead of counting every tied
        # candidate as a full independent win.
        votes = (winners / winners.sum(axis=1, keepdims=True)).mean(axis=0)
        selected = _maximum(votes)
    else:
        raise ValueError(f"Unknown selector {method}")
    return {"selected": selected, "external_leader": leader,
            "eligible_count": int(eligible.sum()) if method in {"M2", "M5", "M6"} else None,
            "leader_changed": bool(selected != leader),
            "selected_gap": float(delta[selected]),
            "selected_bootstrap_sd": None if sigma is None else float(sigma[selected])}
