from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

from .config import ModelConfig, TrainConfig


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for model training. Install `torch` first.")


def pick_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    _require_torch()
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class Normalizer:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray

    def norm_x(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean) / np.clip(self.x_std, 1e-8, None)

    def denorm_x(self, x: np.ndarray) -> np.ndarray:
        return x * np.clip(self.x_std, 1e-8, None) + self.x_mean

    def norm_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_mean) / np.clip(self.y_std, 1e-8, None)

    def denorm_y(self, y: np.ndarray) -> np.ndarray:
        return y * np.clip(self.y_std, 1e-8, None) + self.y_mean


if nn is not None:

    class ConditionalVAE(nn.Module):
        def __init__(self, dim_x: int, dim_y: int, hidden_dim: int, latent_dim: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(dim_x + dim_y, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.mu_head = nn.Linear(hidden_dim, latent_dim)
            self.logvar_head = nn.Linear(hidden_dim, latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim + dim_y, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim_x),
            )

        def encode(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            h = self.encoder(torch.cat([x, y], dim=-1))
            return self.mu_head(h), self.logvar_head(h)

        def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return self.decoder(torch.cat([z, y], dim=-1))

        def forward(
            self, x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            mu, logvar = self.encode(x, y)
            z = self.reparameterize(mu, logvar)
            recon = self.decode(z, y)
            return recon, mu, logvar


    class ConditionalRegressor(nn.Module):
        def __init__(self, dim_x: int, dim_y: int, hidden_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim_y, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim_x),
            )

        def forward(self, y: torch.Tensor) -> torch.Tensor:
            return self.net(y)


@dataclass
class TrainHistory:
    train_loss: list[float]
    val_loss: list[float]


def _build_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    _require_torch()
    ds = TensorDataset(
        torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _kl_div(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def train_cvae(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
) -> tuple["ConditionalVAE", TrainHistory]:
    _require_torch()
    device = pick_device(train_cfg.device)
    model = ConditionalVAE(
        dim_x=x_train.shape[1],
        dim_y=y_train.shape[1],
        hidden_dim=model_cfg.hidden_dim,
        latent_dim=model_cfg.latent_dim,
    ).to(device)
    optim = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )

    train_loader = _build_loader(x_train, y_train, train_cfg.batch_size, shuffle=True)
    val_loader = _build_loader(x_val, y_val, train_cfg.batch_size, shuffle=False)
    history = TrainHistory(train_loss=[], val_loss=[])

    best_val = float("inf")
    best_state = None

    patience = 0
    for _ in range(train_cfg.epochs):
        model.train()
        total = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            recon, mu, logvar = model(xb, yb)
            rec_loss = torch.mean((recon - xb) ** 2)
            kl = _kl_div(mu, logvar)
            loss = rec_loss + model_cfg.beta * kl
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total += float(loss.item()) * xb.shape[0]
            n += xb.shape[0]
        history.train_loss.append(total / max(1, n))

        model.eval()
        vtotal = 0.0
        vn = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                recon, mu, logvar = model(xb, yb)
                rec_loss = torch.mean((recon - xb) ** 2)
                kl = _kl_div(mu, logvar)
                vloss = rec_loss + model_cfg.beta * kl
                vtotal += float(vloss.item()) * xb.shape[0]
                vn += xb.shape[0]
        val_loss = vtotal / max(1, vn)
        history.val_loss.append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= train_cfg.early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def train_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
) -> tuple["ConditionalRegressor", TrainHistory]:
    _require_torch()
    device = pick_device(train_cfg.device)
    model = ConditionalRegressor(
        dim_x=x_train.shape[1], dim_y=y_train.shape[1], hidden_dim=model_cfg.hidden_dim
    ).to(device)
    optim = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )

    train_loader = _build_loader(x_train, y_train, train_cfg.batch_size, shuffle=True)
    val_loader = _build_loader(x_val, y_val, train_cfg.batch_size, shuffle=False)
    history = TrainHistory(train_loss=[], val_loss=[])

    best_val = float("inf")
    best_state = None

    patience = 0
    for _ in range(train_cfg.epochs):
        model.train()
        total = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(yb)
            loss = torch.mean((pred - xb) ** 2)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total += float(loss.item()) * xb.shape[0]
            n += xb.shape[0]
        history.train_loss.append(total / max(1, n))

        model.eval()
        vtotal = 0.0
        vn = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(yb)
                vloss = torch.mean((pred - xb) ** 2)
                vtotal += float(vloss.item()) * xb.shape[0]
                vn += xb.shape[0]
        val_loss = vtotal / max(1, vn)
        history.val_loss.append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= train_cfg.early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def sample_cvae(
    model: "ConditionalVAE",
    y_query: np.ndarray,
    k: int,
    normalizer: Normalizer,
    seed: int,
    device: str = "cpu",
) -> np.ndarray:
    _require_torch()
    model.eval()
    rng = np.random.default_rng(seed)
    y_std = normalizer.norm_y(y_query.reshape(1, -1)).astype(np.float32)
    y = torch.from_numpy(np.repeat(y_std, k, axis=0)).to(device)
    z = torch.from_numpy(rng.normal(size=(k, model.mu_head.out_features)).astype(np.float32)).to(device)
    with torch.no_grad():
        x_std = model.decode(z, y).cpu().numpy()
    return normalizer.denorm_x(x_std)


def sample_regressor(
    model: "ConditionalRegressor",
    y_query: np.ndarray,
    k: int,
    normalizer: Normalizer,
    noise_std: float,
    seed: int,
    device: str = "cpu",
) -> np.ndarray:
    _require_torch()
    model.eval()
    y_std = normalizer.norm_y(y_query.reshape(1, -1)).astype(np.float32)
    y = torch.from_numpy(y_std).to(device)
    with torch.no_grad():
        center = model(y).cpu().numpy()[0]

    rng = np.random.default_rng(seed)
    samples = center[None, :] + rng.normal(scale=noise_std, size=(k, center.shape[0])).astype(np.float32)
    return normalizer.denorm_x(samples)
