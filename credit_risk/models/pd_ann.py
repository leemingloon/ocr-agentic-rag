"""
PyTorch ANN for PD (Probability of Default) — sklearn-style interface.

Architecture: Input → LayerNorm → Dropout(0.1) → [150, 150, 150] (ReLU, LayerNorm, Dropout(0.1)) → sigmoid.
BCE loss, Adam lr=1e-3; early stopping on validation AUC (patience=3).
Used in 02a_pd_xgboost_training.ipynb for comparison with XGBoost/LightGBM.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, List, Any, Union

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _to_tensor(X: np.ndarray, y: Optional[np.ndarray] = None, device: Optional["torch.device"] = None):
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for PD ANN. Install with: pip install torch")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    if y is not None:
        y_t = torch.as_tensor(y, dtype=torch.float32, device=device).unsqueeze(1)
        return X_t, y_t, device
    return X_t, device


class _PDAnnModule(nn.Module):
    """Input → LayerNorm → Dropout → [150,150,150] (ReLU, LayerNorm, Dropout) → sigmoid. LayerNorm avoids BatchNorm's need for >1 sample per batch."""

    def __init__(self, n_features: int, hidden: List[int] = (150, 150, 150), dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        layers: List[nn.Module] = [
            nn.LayerNorm(n_features),
            nn.Dropout(dropout),
        ]
        prev = n_features
        for h in hidden:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(inplace=True),
                nn.LayerNorm(h),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return torch.sigmoid(self.net(x))


class PDAnnClassifier:
    """
    Sklearn-style binary classifier for PD using a small PyTorch ANN.
    fit(X, y, X_val=None, y_val=None) with early stopping on validation AUC.
    After fit, history_ contains train_auc_list and val_auc_list per epoch for learning curves.
    scale_pos_weight: optional class weight for imbalance (float or "auto" = n_neg/n_pos), same idea as XGBoost scale_pos_weight.
    """

    def __init__(
        self,
        hidden: tuple = (150, 150, 150),
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        epochs: int = 20,
        batch_size: int = 32,
        early_stopping_patience: int = 3,
        random_state: Optional[int] = None,
        scale_pos_weight: Optional[Union[float, str]] = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")
        self.hidden = hidden
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.scale_pos_weight = scale_pos_weight
        self._model: Optional[_PDAnnModule] = None
        self._device: Optional[torch.device] = None
        self.n_features_: Optional[int] = None
        self.history_: Dict[str, List[float]] = {"train_auc": [], "val_auc": []}

    def _get_model(self) -> _PDAnnModule:
        if self._model is None or self.n_features_ is None:
            raise RuntimeError("Model not fitted yet.")
        return self._model

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "PDAnnClassifier":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64).ravel()
        if X_val is not None:
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.float64).ravel()
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        self.n_features_ = X.shape[1]
        self._model = _PDAnnModule(self.n_features_, list(self.hidden), self.dropout).to(device)
        self.history_ = {"train_auc": [], "val_auc": []}
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        pos_weight_val = self.scale_pos_weight
        if pos_weight_val == "auto":
            n_pos = float(np.sum(y == 1))
            n_neg = float(np.sum(y == 0))
            pos_weight_val = (n_neg / n_pos) if n_pos > 0 else 1.0
        criterion = nn.BCELoss(reduction="none")
        loader = DataLoader(
            TensorDataset(*_to_tensor(X, y, device)[:2]),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,  # avoid batch size 1 → BatchNorm needs >1 sample per channel
        )
        best_val_auc = -1.0
        best_state: Optional[Dict[str, Any]] = None
        patience_counter = 0
        for epoch in range(self.epochs):
            self._model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                out = self._model(xb)
                el = criterion(out, yb)
                if pos_weight_val is not None and pos_weight_val != 1.0:
                    w = torch.where(yb == 1, torch.full_like(yb, float(pos_weight_val), device=el.device), torch.ones_like(yb, device=el.device))
                    loss = (w * el).mean()
                else:
                    loss = el.mean()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                self._model.eval()
                p_train = self._model(_to_tensor(X, device=device)[0]).cpu().numpy().ravel()
                train_auc = _auc(y, p_train)
                self.history_["train_auc"].append(train_auc)
                if X_val is not None and y_val is not None:
                    p_val = self._model(_to_tensor(X_val, device=device)[0]).cpu().numpy().ravel()
                    val_auc = _auc(y_val, p_val)
                    self.history_["val_auc"].append(val_auc)
                    if val_auc > best_val_auc:
                        best_val_auc = val_auc
                        best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        if best_state is not None:
                            self._model.load_state_dict(best_state)
                            self._model.to(device)
                        break
                else:
                    self.history_["val_auc"].append(train_auc)
        if best_state is not None and X_val is not None:
            self._model.load_state_dict(best_state)
            self._model.to(device)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} features, got {X.shape[1]}")
        self._get_model().eval()
        with torch.no_grad():
            p = self._model(_to_tensor(X, device=self._device)[0]).cpu().numpy()
        p = p.ravel()
        return np.column_stack([1 - p, p])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(np.int64)


def _auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if np.unique(y_true).size < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_prob))
