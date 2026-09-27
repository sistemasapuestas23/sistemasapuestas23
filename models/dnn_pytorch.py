import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from config.settings import DNN_HIDDEN, RANDOM_SEED


class DNNModel(nn.Module):
    def __init__(self, input_dim: int, hidden: list = None, num_classes: int = 3):
        super().__init__()
        hidden = hidden or DNN_HIDDEN
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden: list = None, epochs: int = 50, batch_size: int = 64, lr: float = 1e-3):
        self.hidden = hidden or DNN_HIDDEN
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model_ = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DNNClassifier":
        torch.manual_seed(RANDOM_SEED)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model_ = DNNModel(X.shape[1], self.hidden).to(self.device)
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
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model_(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
