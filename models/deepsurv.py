import torch
import torch.nn as nn
import numpy as np
import os


class DeepSurv(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def negative_partial_log_likelihood(log_hazard, durations, events):
    """Cox partial log-likelihood loss — the core of survival DL."""
    order   = torch.argsort(durations, descending=True)
    log_hz  = log_hazard[order]
    events  = events[order].float()
    log_cum = torch.logcumsumexp(log_hz, dim=0)
    loss    = -((log_hz - log_cum) * events).sum() / events.sum()
    return loss


def train_deepsurv(X_pre, y, epochs=50, lr=1e-3, val_fraction=0.1):
    X_pre = np.ascontiguousarray(X_pre)
    y_dur = np.ascontiguousarray(y["duration"])
    y_ev  = np.ascontiguousarray(y["event"].astype(float))

    n     = len(X_pre)
    n_val = max(1, int(n * val_fraction))
    idx   = np.random.RandomState(42).permutation(n)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train = torch.tensor(X_pre[train_idx], dtype=torch.float32)
    X_val   = torch.tensor(X_pre[val_idx],   dtype=torch.float32)
    dur_train = torch.tensor(y_dur[train_idx], dtype=torch.float32)
    dur_val   = torch.tensor(y_dur[val_idx],   dtype=torch.float32)
    ev_train  = torch.tensor(y_ev[train_idx],  dtype=torch.float32)
    ev_val    = torch.tensor(y_ev[val_idx],    dtype=torch.float32)

    model = DeepSurv(input_dim=X_pre.shape[1])
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        train_loss = negative_partial_log_likelihood(model(X_train), dur_train, ev_train)
        train_loss.backward()
        opt.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = negative_partial_log_likelihood(model(X_val), dur_val, ev_val)
            print(f"Epoch {epoch+1}/{epochs}  train={train_loss.item():.4f}  val={val_loss.item():.4f}")

    model_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "deepsurv.pt"))
    return model


def predict_risk(model, X_pre):
    model.eval()
    with torch.no_grad():
        X_pre = np.ascontiguousarray(X_pre)
        X_t = torch.tensor(X_pre, dtype=torch.float32)
        return model(X_t).numpy()
