import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from config.settings import LSTM_HIDDEN, LSTM_LAYERS, LSTM_MOMENTUM_WINDOW, RANDOM_SEED


class LSTMMomentumModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: int = LSTM_HIDDEN,
        layers: int = LSTM_LAYERS,
        num_classes: int = 3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden,
            layers,
            batch_first=True,
            dropout=0.2 if layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h_n, _) = self.lstm(x)
        momentum = h_n[-1] - (h_n[-2] if h_n.shape[0] > 1 else h_n[-1])
        combined = out[:, -1, :] + 0.3 * momentum
        return self.fc(combined)


class LSTMMomentumClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        window: int = LSTM_MOMENTUM_WINDOW,
        hidden: int = LSTM_HIDDEN,
        layers: int = LSTM_LAYERS,
        epochs: int = 40,
        batch_size: int = 64,
        lr: float = 1e-3,
    ):
        self.window = window
        self.hidden = hidden
        self.layers = layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model_ = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_sequences(self, X: np.ndarray) -> np.ndarray:
        n, f = X.shape
        if n < self.window:
            return X[:, np.newaxis, :]
        seqs = []
        for i in range(n):
            start = max(0, i - self.window + 1)
            seq = X[start : i + 1]
            if len(seq) < self.window:
                pad = np.zeros((self.window - len(seq), f))
                seq = np.vstack([pad, seq])
            seqs.append(seq)
        return np.stack(seqs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMMomentumClassifier":
        torch.manual_seed(RANDOM_SEED)
        X_seq = self._create_sequences(X)
        X_t = torch.tensor(X_seq, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model_ = LSTMMomentumModel(X.shape[1], self.hidden, self.layers).to(
            self.device
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                out = self.model_(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model_.eval()
        with torch.no_grad():
            X_seq = self._create_sequences(X)
            X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
            logits = self.model_(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
