import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from config.settings import LSTM_HIDDEN, LSTM_LAYERS, RANDOM_SEED


class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden: int = LSTM_HIDDEN, layers: int = LSTM_LAYERS, num_classes: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden, layers, batch_first=True, dropout=0.2 if layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LSTMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self, hidden: int = LSTM_HIDDEN, layers: int = LSTM_LAYERS,
        epochs: int = 40, batch_size: int = 64, lr: float = 1e-3
    ):
        self.hidden = hidden
        self.layers = layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model_ = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMClassifier":
        torch.manual_seed(RANDOM_SEED)
        X_seq = X[:, np.newaxis, :]
        X_t = torch.tensor(X_seq, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model_ = LSTMModel(X.shape[1], self.hidden, self.layers).to(self.device)
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
            X_seq = X[:, np.newaxis, :]
            X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
            logits = self.model_(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
