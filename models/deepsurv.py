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
            nn.Linear(32, 1),   # outputs log-hazard score
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


def train_deepsurv(X_pre, y, epochs=50, lr=1e-3):
    X_pre = np.ascontiguousarray(X_pre)

    X_t   = torch.tensor(X_pre, dtype=torch.float32)
    dur_t = torch.tensor(
        np.ascontiguousarray(y["duration"]),
        dtype=torch.float32
        )

    ev_t = torch.tensor(
        np.ascontiguousarray(y["event"].astype(float)),
        dtype=torch.float32
        )

    model = DeepSurv(input_dim=X_pre.shape[1])
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        opt.zero_grad()

        loss = negative_partial_log_likelihood(
            model(X_t), dur_t, ev_t
        )

        loss.backward()
        opt.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}  loss={loss.item():.4f}")

    model_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(model_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(model_dir, "deepsurv.pt"))
    return model


def predict_risk(model, X_pre):
    model.eval()
    with torch.no_grad():
        # Ensure array is contiguous to avoid numpy stride errors
        X_pre = np.ascontiguousarray(X_pre)
        X_t = torch.tensor(X_pre, dtype=torch.float32)
        return model(X_t).numpy()