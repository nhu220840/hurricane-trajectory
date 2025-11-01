# src/evaluate.py

import pickle
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from .config import (
    PROCESSED_NPZ, SCALER_Y_PKL,
    CHECKPOINT_LSTM_TORCH, CHECKPOINT_LSTM_SCRATCH,
    LSTM_SCRATCH, LSTM_TORCH  # (NEW) Import LSTM_TORCH for consistency
)
from .models import LSTMForecaster, LSTMFromScratchForecaster


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers
    from math import radians, sin, cos, asin, sqrt
    p = radians
    dlat = p(lat2 - lat1)
    dlon = p(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(p(lat1)) * cos(p(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


def _load_test_split():
    """
    (REVISED) Loads the pre-split test arrays from the .npz file.
    """
    print(f"[Load] Loading pre-split test data from {PROCESSED_NPZ}...")
    try:
        data = np.load(PROCESSED_NPZ, allow_pickle=True)
        X_test = data["X_test"]
        Y_test = data["Y_test"]
        last_test = data["last_obs_test"]
        print(f"[Load] Test data shape: {X_test.shape}")
    except FileNotFoundError:
        print(f"ERROR: File not found: {PROCESSED_NPZ}")
        print("Please run the data processing step first (e.g., python main.py --process-data)")
        return None, None, None  # Return None to fail gracefully
    except KeyError as e:
        print(f"ERROR: Missing expected array {e} in {PROCESSED_NPZ}.")
        print("The .npz file might be old or corrupted. Please re-run data processing.")
        return None, None, None  # Return None to fail gracefully

    return X_test, Y_test, last_test


# (REMOVED) _split_by_sid - No longer needed, splitting is done in data_processing.py
# (REMOVED) _filter_by_sid_idx - No longer needed


def _safe_load_state_dict(ckpt_path: Path, device: str):
    # (This function is unchanged)
    try:
        # Try loading state_dict (if ckpt saved state_dict)
        state = torch.load(ckpt_path, map_location=device)
        # Check if this is a state_dict or a full (old) model
        if not isinstance(state, dict) or "model_state_dict" not in state:
            # This is a raw state_dict file from the original repo
            return state

        # This is a new checkpoint file
        if "model_state_dict" in state:
            return state["model_state_dict"]
        else:
            return state  # Fallback

    except Exception as e:
        print(f"Error loading checkpoint {ckpt_path}: {e}")
        # Try fallback loading the full model (less safe)
        try:
            state = torch.load(ckpt_path, map_location=device, pickle_module=pickle)
            if hasattr(state, 'state_dict'):  # If it's a model
                return state.state_dict()
            return state  # Return whatever was loaded
        except Exception as e2:
            print(f"Fallback loading failed: {e2}")
            return None


def _predict_errs_km_and_deltas(model, ckpt_path: Path,
                                X_test: np.ndarray, Y_test: np.ndarray, last_obs: np.ndarray,
                                scaler_y, device: str):
    """
    Returns:
      errs_km: (B,)
      y_pred_deg: (B,2) predicted delta (degrees) after inverse scaling
      y_true_deg: (B,2) true delta (degrees)
      lat_true, lon_true, lat_pred, lon_pred: (B,)
    """
    # (This function is unchanged, but with added safety checks)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return None

    try:
        state_dict = _safe_load_state_dict(ckpt_path, device=device)
        if state_dict is None:
            print(f"Failed to load state dict from {ckpt_path}")
            return None
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Error loading state_dict for model {model.__class__.__name__}: {e}")
        print("Checkpoint might be incompatible with model architecture.")
        return None
    except Exception as e:
        print(f"Unknown error loading checkpoint {ckpt_path}: {e}")
        return None

    model.to(device)
    model.eval()

    with torch.no_grad():
        y_pred_scaled = model(torch.tensor(X_test, dtype=torch.float32, device=device)).cpu().numpy()

    y_pred_deg = scaler_y.inverse_transform(y_pred_scaled)  # (B,2) predicted delta (degrees)
    y_true_deg = scaler_y.inverse_transform(Y_test)  # (B,2) true delta (degrees)

    lat_true = last_obs[:, 0] + y_true_deg[:, 0]
    lon_true = last_obs[:, 1] + y_true_deg[:, 1]
    lat_pred = last_obs[:, 0] + y_pred_deg[:, 0]
    lon_pred = last_obs[:, 1] + y_pred_deg[:, 1]

    errs_km = np.array([
        haversine_km(la_t, lo_t, la_p, lo_p)
        for la_t, lo_t, la_p, lo_p in zip(lat_true, lon_true, lat_pred, lon_pred)
    ])
    return errs_km, y_pred_deg, y_true_deg, lat_true, lon_true, lat_pred, lon_pred


def _summary_and_print(name: str, errs_km: np.ndarray, y_pred_deg: np.ndarray, y_true_deg: np.ndarray):
    # (This function is unchanged)
    mae_km = float(np.mean(np.abs(errs_km)))
    mse_km = float(np.mean(errs_km ** 2))
    mae_deg = float(np.mean(np.abs(y_pred_deg - y_true_deg)))
    mse_deg = float(np.mean((y_pred_deg - y_true_deg) ** 2))
    p50 = float(np.percentile(errs_km, 50))
    p75 = float(np.percentile(errs_km, 75))
    p90 = float(np.percentile(errs_km, 90))
    print(f"[{name}] MAE_km={mae_km:.3f} | MSE_km={mse_km:.3f} | "
          f"MAE_deg={mae_deg:.5f} | MSE_deg={mse_deg:.5f} | "
          f"P50={p50:.2f}km | P75={p75:.2f}km | P90={p90:.2f}km")
    return {"name": name, "mae_km": mae_km, "mse_km": mse_km,
            "mae_deg": mae_deg, "mse_deg": mse_deg,
            "p50_km": p50, "p75_km": p75, "p90_km": p90}


def _plot_ecdf(errs_dict, out_png: Path):
    """
    ECDF: x = error (km), y = proportion ≤ x
    (This function is unchanged)
    """
    plt.figure()
    for label, errs in errs_dict.items():
        if errs is None:
            continue
        e = np.sort(errs)
        y = np.arange(1, len(e) + 1) / len(e)
        plt.plot(e, y, label=label)
    plt.xlabel("Error (km)")
    plt.ylabel("Proportion ≤ error")
    plt.title("ECDF of per-sample great-circle error (km) – test set")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {out_png}")


def _plot_sorted(errs_dict, out_png: Path):
    """
    Sorted error: x = sample rank (asc by error), y = error (km)
    (This function is unchanged)
    """
    plt.figure()
    for label, errs in errs_dict.items():
        if errs is None:
            continue
        e = np.sort(errs)
        x = np.arange(len(e))
        plt.plot(x, e, label=label)
    plt.xlabel("Sample rank (ascending by error)")
    plt.ylabel("Error (km)")
    plt.title("Sorted per-sample error (km) – test set")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {out_png}")


def evaluate():
    print("[Evaluate] Starting evaluation...")
    # Load test data
    X_test, Y_test, last_test = _load_test_split()
    if X_test is None:
        print("[Evaluate] Could not load test data. Aborting evaluation.")
        return None

    try:
        with open(SCALER_Y_PKL, "rb") as f:
            scaler_y = pickle.load(f)
        print(f"[Load] Loaded Y scaler from {SCALER_Y_PKL}")
    except FileNotFoundError:
        print(f"ERROR: Scaler file not found: {SCALER_Y_PKL}. Aborting.")
        return None

    input_size = X_test.shape[-1]
    out_dim = Y_test.shape[-1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Evaluate] Using device: {device}")

    results = {}

    # --- PyTorch model ---
    errs_torch = None
    if CHECKPOINT_LSTM_TORCH.exists():
        print(f"[Evaluate] Loading model 'pytorch' from {CHECKPOINT_LSTM_TORCH}")
        m_torch = LSTMForecaster(
            input_size=input_size,
            hidden_size=LSTM_TORCH["hidden_size"],
            num_layers=LSTM_TORCH["num_layers"],
            dropout=LSTM_TORCH["dropout"]
        )
        out = _predict_errs_km_and_deltas(m_torch, CHECKPOINT_LSTM_TORCH,
                                          X_test, Y_test, last_test, scaler_y, device)
        if out is not None:
            errs_torch, y_pred_deg_t, y_true_deg_t, lat_true_t, lon_true_t, lat_pred_t, lon_pred_t = out
            results["pytorch"] = _summary_and_print("pytorch", errs_torch, y_pred_deg_t, y_true_deg_t)
    else:
        print(f"[pytorch] Skipping: checkpoint not found at {CHECKPOINT_LSTM_TORCH}")

    # --- Scratch model ---
    errs_scratch = None
    if CHECKPOINT_LSTM_SCRATCH.exists():
        print(f"[Evaluate] Loading model 'scratch' from {CHECKPOINT_LSTM_SCRATCH}")
        m_scratch = LSTMFromScratchForecaster(
            in_dim=input_size,
            hidden=LSTM_SCRATCH["hidden_size"],
            num_layers=LSTM_SCRATCH["num_layers"],
            out_dim=out_dim,
            dropout=LSTM_SCRATCH["dropout"]
        )
        out = _predict_errs_km_and_deltas(m_scratch, CHECKPOINT_LSTM_SCRATCH,
                                          X_test, Y_test, last_test, scaler_y, device)
        if out is not None:
            errs_scratch, y_pred_deg_s, y_true_deg_s, lat_true_s, lon_true_s, lat_pred_s, lon_pred_s = out
            results["scratch"] = _summary_and_print("scratch", errs_scratch, y_pred_deg_s, y_true_deg_s)
    else:
        print(f"[scratch] Skipping: checkpoint not found at {CHECKPOINT_LSTM_SCRATCH}")

    # --- Baseline (persistence Δ=0) ---
    print("[Evaluate] Calculating baseline (persistence) model...")
    y_true_deg = scaler_y.inverse_transform(Y_test)
    lat_true = last_test[:, 0] + y_true_deg[:, 0]
    lon_true = last_test[:, 1] + y_true_deg[:, 1]
    lat_base = last_test[:, 0]  # Baseline prediction is no change (delta=0)
    lon_base = last_test[:, 1]
    errs_base = np.array([haversine_km(a, b, c, d) for a, b, c, d in zip(lat_true, lon_true, lat_base, lon_base)])

    results["baseline"] = {
        "name": "baseline",
        "mae_km": float(np.mean(np.abs(errs_base))),
        "mse_km": float(np.mean(errs_base ** 2)),
        "p50_km": float(np.percentile(errs_base, 50)),
        "p75_km": float(np.percentile(errs_base, 75)),
        "p90_km": float(np.percentile(errs_base, 90)),
    }
    print("[baseline] "
          f"MAE_km={results['baseline']['mae_km']:.3f} | "
          f"MSE_km={results['baseline']['mse_km']:.3f} | "
          f"P50={results['baseline']['p50_km']:.2f}km | "
          f"P75={results['baseline']['p75_km']:.2f}km | "
          f"P90={results['baseline']['p90_km']:.2f}km")

    # --- Plot easy-to-read charts ---
    print("[Evaluate] Generating plots...")
    plots_dir = Path("results/plots")
    ecdf_png = plots_dir / "compare_ecdf_km.png"
    sorted_png = plots_dir / "compare_sorted_error_km.png"

    _plot_ecdf(
        errs_dict={
            "Baseline (Persistence)": errs_base,
            "LSTM PyTorch": errs_torch,
            "LSTM Scratch": errs_scratch,
        },
        out_png=ecdf_png,
    )

    _plot_sorted(
        errs_dict={
            "Baseline (Persistence)": errs_base,
            "LSTM PyTorch": errs_torch,
            "LSTM Scratch": errs_scratch,
        },
        out_png=sorted_png,
    )

    print("[Evaluate] Evaluation complete.")
    return results