"""
LSTM-based short-term forecaster.

Drop-in alternative to ShortTermModel (LightGBM quantile). Same interface:
  • fit(X, y)        — y can be multi-target (DataFrame with one col per var)
  • predict(X, now)  — returns list[QuantilePrediction]
  • save / load      — pickled state_dict + scaler

Architecture:
  Input  → MinMax scaler
         → 1-layer LSTM (hidden=64) over a sliding window of `lookback` hours
         → Linear head outputting 3 quantiles per target variable
  Loss:  pinball loss summed over (q10, q50, q90) and over targets

Why LSTM not Transformer/TFT:
  • LSTMs are well-suited to short-horizon (1-72h) point forecasting
    when you have <5 years of training data per location. TFT shines
    with much more data and many heterogeneous features per timestep.
  • LSTMs train in seconds on CPU at this scale. TFT is GPU-heavy.
  • The interface contract is what matters — swap LSTM → TFT later by
    changing this one module without touching anything else.

If torch isn't installed, the module imports cleanly but `LSTMModel`
raises ImportError on construction. Everything else in the project keeps
working with LightGBM as the short-term model.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    _TORCH_OK = True
except ImportError:
    torch = None  # type: ignore
    nn = None     # type: ignore
    _TORCH_OK = False

from src.models.short_term import QuantilePrediction

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────── network

if _TORCH_OK:

    class _QuantileLSTM(nn.Module):
        """LSTM with a multi-quantile multi-target output head."""

        def __init__(self, n_features: int, n_targets: int,
                     n_quantiles: int = 3, hidden: int = 64,
                     n_layers: int = 1, dropout: float = 0.1):
            super().__init__()
            self.n_targets = n_targets
            self.n_quantiles = n_quantiles
            self.lstm = nn.LSTM(
                input_size=n_features, hidden_size=hidden,
                num_layers=n_layers, batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            self.head = nn.Linear(hidden, n_targets * n_quantiles)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (batch, seq, n_features)
            out, _ = self.lstm(x)
            last = out[:, -1, :]                 # last timestep
            y = self.head(last)                  # (batch, n_targets * n_quantiles)
            return y.view(-1, self.n_targets, self.n_quantiles)


def _pinball_loss(pred: "torch.Tensor", target: "torch.Tensor",
                  quantiles: "torch.Tensor") -> "torch.Tensor":
    """pred: (B, T, Q), target: (B, T), quantiles: (Q,)"""
    target = target.unsqueeze(-1)                # (B, T, 1)
    diff = target - pred                         # (B, T, Q)
    loss = torch.maximum(quantiles * diff, (quantiles - 1) * diff)
    return loss.mean()


# ─────────────────────────────────────────────────── public model

@dataclass
class _ScalerState:
    mins: np.ndarray
    maxs: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        rng = np.where(self.maxs - self.mins > 1e-9,
                       self.maxs - self.mins, 1.0)
        return (X - self.mins) / rng

    @classmethod
    def fit(cls, X: np.ndarray) -> "_ScalerState":
        return cls(mins=X.min(axis=0), maxs=X.max(axis=0))


class LSTMModel:
    """Quantile LSTM — same interface as ShortTermModel.

    Construction raises ImportError if PyTorch isn't installed; callers
    should fall back to ShortTermModel in that case (and the orchestrator
    does so automatically).
    """

    def __init__(self, *, lookback: int = 48,
                 quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
                 hidden: int = 64, n_layers: int = 1,
                 epochs: int = 30, batch_size: int = 64,
                 lr: float = 1e-3, device: str | None = None,
                 horizon_hours: int = 72):
        if not _TORCH_OK:
            raise ImportError(
                "PyTorch not installed. `pip install torch` to use LSTMModel; "
                "fall back to ShortTermModel (LightGBM) otherwise.")
        self.lookback = lookback
        self.quantiles = quantiles
        self.hidden = hidden
        self.n_layers = n_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.horizon_hours = horizon_hours
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._net: "_QuantileLSTM | None" = None
        self._scaler: _ScalerState | None = None
        self._target_cols: list[str] | None = None
        self._feature_cols: list[str] | None = None

    # --------------------------------------------------------- windowing

    def _make_windows(self, X: np.ndarray, y: np.ndarray | None
                      ) -> tuple[np.ndarray, np.ndarray | None]:
        """Build sliding windows of `lookback` rows. Targets, if given,
        are taken at the row immediately after each window."""
        n = len(X)
        if n <= self.lookback:
            raise ValueError(
                f"need >{self.lookback} rows, got {n}")
        windows = np.stack([X[i: i + self.lookback]
                            for i in range(n - self.lookback)])
        if y is None:
            return windows, None
        targets = y[self.lookback: self.lookback + len(windows)]
        return windows, targets

    # --------------------------------------------------------------- fit

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "LSTMModel":
        self._feature_cols = list(X.columns)
        self._target_cols = list(y.columns)

        X_arr = X.to_numpy(dtype=np.float32)
        y_arr = y.to_numpy(dtype=np.float32)
        self._scaler = _ScalerState.fit(X_arr)
        X_scaled = self._scaler.transform(X_arr).astype(np.float32)

        windows, targets = self._make_windows(X_scaled, y_arr)
        # tensors
        Xt = torch.from_numpy(windows).to(self.device)
        Yt = torch.from_numpy(targets).to(self.device)

        self._net = _QuantileLSTM(
            n_features=X_scaled.shape[1],
            n_targets=y_arr.shape[1],
            n_quantiles=len(self.quantiles),
            hidden=self.hidden,
            n_layers=self.n_layers,
        ).to(self.device)

        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        q_tensor = torch.tensor(self.quantiles, device=self.device,
                                dtype=torch.float32)

        n = len(Xt)
        for epoch in range(self.epochs):
            self._net.train()
            perm = torch.randperm(n, device=self.device)
            losses = []
            for i in range(0, n, self.batch_size):
                idx = perm[i: i + self.batch_size]
                xb, yb = Xt[idx], Yt[idx]
                opt.zero_grad()
                pred = self._net(xb)              # (B, T, Q)
                loss = _pinball_loss(pred, yb, q_tensor)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if epoch == 0 or (epoch + 1) % 10 == 0:
                log.info("LSTM epoch %d/%d  loss=%.4f",
                         epoch + 1, self.epochs, float(np.mean(losses)))
        return self

    # ------------------------------------------------------------ predict

    def predict(self, X: pd.DataFrame,
                now: pd.Timestamp | None = None) -> list[QuantilePrediction]:
        if self._net is None or self._scaler is None:
            raise RuntimeError("model not fitted")
        # Align inference features to training schema.
        X_aligned = X.reindex(columns=self._feature_cols, fill_value=0.0)
        X_aligned = X_aligned.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        if len(X_aligned) <= self.lookback:
            log.warning("LSTM: need >%d input rows for predict, got %d — "
                        "returning empty", self.lookback, len(X_aligned))
            return []

        X_scaled = self._scaler.transform(
            X_aligned.to_numpy(dtype=np.float32)).astype(np.float32)
        windows, _ = self._make_windows(X_scaled, None)
        Xt = torch.from_numpy(windows).to(self.device)

        self._net.eval()
        with torch.no_grad():
            pred = self._net(Xt).cpu().numpy()       # (B, T, Q)

        # Window i predicts the row at index lookback + i
        valid_times = X_aligned.index[self.lookback:
                                      self.lookback + len(windows)]
        ref = (now if now is not None
               else pd.Timestamp.utcnow().tz_localize(None))
        if ref.tzinfo is None and valid_times.tz is not None:
            ref = ref.tz_localize("UTC")
        leads = ((valid_times - ref).total_seconds() / 3600).to_numpy()

        out: list[QuantilePrediction] = []
        for ti, var in enumerate(self._target_cols):
            for wi, t in enumerate(valid_times):
                qvals = pred[wi, ti]      # (Q,)
                # quantiles are stored in the model in self.quantiles order
                q_lookup = dict(zip(self.quantiles, qvals))
                out.append(QuantilePrediction(
                    variable=var, valid_time=t, lead_hours=float(leads[wi]),
                    q10=float(q_lookup[0.1]),
                    q50=float(q_lookup[0.5]),
                    q90=float(q_lookup[0.9]),
                ))
        return out

    # ------------------------------------------------------------- I/O

    def save(self, path: str | Path) -> None:
        if self._net is None:
            raise RuntimeError("nothing to save")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "state_dict": self._net.state_dict(),
            "scaler": self._scaler,
            "feature_cols": self._feature_cols,
            "target_cols": self._target_cols,
            "config": {
                "lookback": self.lookback,
                "quantiles": self.quantiles,
                "hidden": self.hidden,
                "n_layers": self.n_layers,
                "horizon_hours": self.horizon_hours,
            },
        }
        with open(path, "wb") as f:
            pickle.dump(blob, f)

    @classmethod
    def load(cls, path: str | Path) -> "LSTMModel":
        if not _TORCH_OK:
            raise ImportError("PyTorch required to load LSTMModel")
        with open(path, "rb") as f:
            blob = pickle.load(f)
        cfg = blob["config"]
        m = cls(lookback=cfg["lookback"], quantiles=cfg["quantiles"],
                hidden=cfg["hidden"], n_layers=cfg["n_layers"],
                horizon_hours=cfg["horizon_hours"])
        m._scaler = blob["scaler"]
        m._feature_cols = blob["feature_cols"]
        m._target_cols = blob["target_cols"]
        m._net = _QuantileLSTM(
            n_features=len(m._feature_cols),
            n_targets=len(m._target_cols),
            n_quantiles=len(m.quantiles),
            hidden=m.hidden, n_layers=m.n_layers,
        ).to(m.device)
        m._net.load_state_dict(blob["state_dict"])
        m._net.eval()
        return m
