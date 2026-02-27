"""
Quantum PD (Probability of Default) model for eval_runner.

Loads the VQC artifact saved by notebook 03_pd_quantum_training.ipynb
(models/pd/pd_quantum_vqc_v1.pkl) and exposes predict_pd(features) compatible
with the same feature dict as PDModel (15 LendingClub features). Uses the
saved 6 quantum features + scaler + params to run the circuit on CPU (PennyLane default.qubit).
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import joblib

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False


def _build_circuit_baseline(n_qubits: int):
    """Build baseline VQC circuit (RY encoding + 2*n_qubits params)."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(params, x):
        for i in range(n_qubits):
            qml.RY(x[i] * np.pi, wires=i)
        for i in range(n_qubits):
            qml.RZ(params[i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        for i in range(n_qubits):
            qml.RZ(params[n_qubits + i], wires=i)
        return qml.expval(qml.PauliZ(0))

    return circuit


def _build_circuit_improved(n_qubits: int, n_params: int):
    """Build improved VQC circuit (data re-upload, 4*n_qubits per layer)."""
    n_reupload = n_params // (4 * n_qubits) if n_qubits else 1
    n_params_per_layer = 4 * n_qubits
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, diff_method="backprop")
    def circuit(params, x):
        for layer in range(n_reupload):
            for i in range(n_qubits):
                qml.RY(x[i] * np.pi, wires=i)
                qml.RZ(x[i] * np.pi, wires=i)
            off = layer * n_params_per_layer
            for i in range(n_qubits):
                qml.RY(params[off + i], wires=i)
                qml.RZ(params[off + n_qubits + i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            for i in range(n_qubits):
                qml.RY(params[off + 2 * n_qubits + i], wires=i)
                qml.RZ(params[off + 3 * n_qubits + i], wires=i)
        return qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(n_qubits)]) / n_qubits)

    return circuit


class QuantumPDModel:
    """
    Quantum VQC-based PD model. Load from pd_quantum_vqc_v1.pkl and call predict_pd(features).
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._params = None
        self._circuit = None
        self._scaler = None
        self._feature_names = None  # 6 quantum feature names in order
        self._all_feature_names = None  # 15 from common_features

    def load(self, filepath: str) -> None:
        if not PENNYLANE_AVAILABLE:
            raise RuntimeError("PennyLane is required for QuantumPDModel. pip install pennylane")
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Quantum PD artifact not found: {path}")
        data = joblib.load(path)
        self._params = np.asarray(data["params"])
        n_qubits = int(data["n_qubits"])
        circuit_type = data.get("circuit_type", "baseline")
        self._scaler = data["scaler"]
        self._feature_names = list(data["feature_names"])

        from credit_risk.feature_engineering.common_features import get_feature_names
        self._all_feature_names = get_feature_names()

        if circuit_type == "improved":
            self._circuit = _build_circuit_improved(n_qubits, len(self._params))
        else:
            self._circuit = _build_circuit_baseline(n_qubits)
        self.model_path = str(path)

    def predict_pd(self, features: Dict[str, float]) -> float:
        """
        Predict default probability from a feature dict (same format as PDModel).
        Builds full 15-dim feature vector, scales (scaler was fit on 15 dims), then
        takes the 6 quantum features in order and runs the VQC circuit.
        """
        if self._circuit is None or self._params is None or self._scaler is None:
            return 0.0
        # Full 15-dim vector in get_feature_names() order (scaler expects this)
        full_vec = np.array([
            float(features.get(name, 0.0))
            for name in self._all_feature_names
        ], dtype=np.float64)
        full_vec = np.nan_to_num(full_vec, nan=0.0, posinf=0.0, neginf=0.0)
        full_vec = full_vec.reshape(1, -1)
        scaled_full = self._scaler.transform(full_vec)
        scaled_full = np.nan_to_num(scaled_full, nan=0.0, posinf=0.0, neginf=0.0)
        # Indices of the 6 quantum features in the 15-dim list
        name_to_idx = {n: i for i, n in enumerate(self._all_feature_names)}
        idx_6 = [name_to_idx[n] for n in self._feature_names if n in name_to_idx]
        if len(idx_6) != len(self._feature_names):
            return 0.0
        x = scaled_full[0][idx_6]
        out = self._circuit(self._params, x)
        out = float(out)
        # Map [-1, 1] -> [0, 1]
        prob = (out + 1.0) / 2.0
        return float(np.clip(prob, 0.0, 1.0))
