# Functions used in MLP dataset processing, model training and testing, etc.
# --- Standard library
import time
import math
import random

# --- Core scientific stack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- PyTorch
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# --- Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#### Split dataset ####
def split_prop_param(
    prop,
    param,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_state=0,
    print_summary=True,
):
    """
    Split `prop` and `param` into train/validation/test sets.

    Parameters
    ----------
    prop : array-like
        Target/properties array.
    param : array-like
        Input/parameter array.
    train_ratio : float
        Fraction of data for training.
    val_ratio : float
        Fraction of data for validation.
    test_ratio : float
        Fraction of data for testing.
    random_state : int
        Random seed for reproducibility.
    verbose : bool
        If True, print a summary of the split.

    Returns
    -------
    prop_train, prop_val, prop_test, param_train, param_val, param_test
    """
    # Optional: normalize ratios so they sum to 1 (in case of small numerical issues)
    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio   /= total
    test_ratio  /= total

    # --- First split: train vs temp (val + test)
    prop_train, prop_temp, param_train, param_temp = train_test_split(
        prop,
        param,
        test_size=(1 - train_ratio),
        random_state=random_state,
    )

    # --- Then split temp into validation and test
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    prop_val, prop_test, param_val, param_test = train_test_split(
        prop_temp,
        param_temp,
        test_size=relative_test_ratio,
        random_state=random_state,
    )

    if print_summary:
        print("=== Dataset Split ===")
        print(f"Train: {prop_train.shape[0]} samples")
        print(f"Val:   {prop_val.shape[0]} samples")
        print(f"Test:  {prop_test.shape[0]} samples")

    return prop_train, prop_val, prop_test, param_train, param_val, param_test


#### scale dataset ####
def scale_datasets(
    prop_train,
    prop_val,
    prop_test,
    param_train,
    param_val,
    param_test,
    feature_range=(-1, 1),
    scale_inputs: bool = True,
    scale_outputs: bool = True,
    print_summary: bool = True,
):
    """
    Scale input (prop_*) and output (param_*) datasets with MinMaxScaler.

    Returns:
        prop_train_scaled, prop_val_scaled, prop_test_scaled,
        param_train_scaled, param_val_scaled, param_test_scaled,
        in_scaler, out_scaler
    """
    in_scaler = None
    out_scaler = None

    # Inputs
    if scale_inputs:
        in_scaler = MinMaxScaler(feature_range=feature_range)
        prop_train_scaled = in_scaler.fit_transform(prop_train)
        prop_val_scaled   = in_scaler.transform(prop_val)
        prop_test_scaled  = in_scaler.transform(prop_test)
    else:
        prop_train_scaled, prop_val_scaled, prop_test_scaled = prop_train, prop_val, prop_test

    # Outputs
    if scale_outputs:
        out_scaler = MinMaxScaler(feature_range=feature_range)
        param_train_scaled = out_scaler.fit_transform(param_train)
        param_val_scaled   = out_scaler.transform(param_val)
        param_test_scaled  = out_scaler.transform(param_test)
    else:
        param_train_scaled, param_val_scaled, param_test_scaled = param_train, param_val, param_test

    if print_summary:
        print("=== Scaling Info ===")
        print(f"Inputs scaled:  {scale_inputs}")
        print(f"Outputs scaled: {scale_outputs}")
        print(f"Feature range:  {feature_range}")

    return (
        prop_train_scaled,
        prop_val_scaled,
        prop_test_scaled,
        param_train_scaled,
        param_val_scaled,
        param_test_scaled,
        in_scaler,
        out_scaler,
    )


#### MLP model class ####
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(128, 64), out_dim=6, dropout=0.0):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    
#### count parameter numbers of model ####
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#### plot MLP schematic ####
def plot_mlp_schematic(input_size, hidden_sizes, output_size, node_scale=0.08):
    layer_sizes = [input_size] + list(hidden_sizes) + [output_size]

    # x positions for layers
    xs = np.linspace(0, 1, num=len(layer_sizes))
    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.axis("off")

    # draw nodes
    for li, size in enumerate(layer_sizes):
        ys = np.linspace(0.1, 0.9, num=size)
        for y in ys:
            circ = plt.Circle((xs[li], y), node_scale, fill=False)
            ax.add_patch(circ)
        
        # label layer
        if li == 0:
            label = f"Input\n({size})"
        elif li == len(layer_sizes)-1:
            label = f"Output\n({size})"
        else:
            label = f"Hidden {li}\n({size})"
        ax.text(xs[li], 0.03, label, ha="center", va="center")

    # draw edges between layers (sparse preview for clarity)
    for li in range(len(layer_sizes)-1):
        size_a, size_b = layer_sizes[li], layer_sizes[li+1]
        ya = np.linspace(0.1, 0.9, num=min(size_a, 10))  # cap to 10 for readability
        yb = np.linspace(0.1, 0.9, num=min(size_b, 10))
        for a in ya:
            for b in yb:
                plt.plot([xs[li]+node_scale, xs[li+1]-node_scale], [a, b], linewidth=0.3)

    plt.title(f"MLP schematic: {layer_sizes}")
    plt.tight_layout()
    plt.show()
    

#### train MLP model ####
def train_model(
    model,
    train_loader,
    val_loader,
    epochs=100,
    patience=20,
    criterion=None,
    optimizer=None,
):
    """
    Train a model with early stopping on validation loss.

    Args:
        model:        nn.Module
        train_loader: DataLoader
        val_loader:   DataLoader
        epochs:       max number of epochs
        patience:     early stopping patience
        criterion:    loss function (e.g. nn.MSELoss())
        optimizer:    optimizer (e.g. torch.optim.Adam(model.parameters(), ...))
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Use provided criterion / optimizer or fall back to defaults
    if criterion is None:
        criterion = nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    print("=== Training started ===")
    for epoch in range(1, epochs + 1):

        # ----- Training -----
        model.train()
        total_train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * xb.size(0)
        train_loss = total_train_loss / len(train_loader.dataset)

        # ----- Validation -----
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                total_val_loss += loss.item() * xb.size(0)
        val_loss = total_val_loss / len(val_loader.dataset)

        # ----- Record and print -----
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f"Epoch {epoch:3d}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        # ----- Early stopping -----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch} (best val = {best_val_loss:.6f})")
                break

    # Restore the best model weights
    if best_state is not None:
        model.load_state_dict(best_state)

    print("=== Training complete ===")
    return model, train_losses, val_losses


#### plot learning curves of training process ####
def plot_learning_curves(train_losses, val_losses, metric_name="MSE Loss"):
    """
    Plot training and validation loss on linear and log scales.

    Args:
        train_losses: list/array of training losses per epoch
        val_losses:   list/array of validation losses per epoch
        metric_name:  label for the y-axis (e.g. 'MSE Loss', 'MAE', etc.)
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Left: linear scale
    axs[0].plot(train_losses, label="Train loss")
    axs[0].plot(val_losses, label="Val loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel(metric_name)
    axs[0].set_title("Learning Curves")
    axs[0].legend()

    # Right: log scale
    axs[1].plot(train_losses, label="Train loss")
    axs[1].plot(val_losses, label="Val loss")
    axs[1].set_yscale("log")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel(f"{metric_name} (log scale)")
    axs[1].set_title("Learning Curves (Log Scale)")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


#### evaluate the model with given loader ####
def evaluate_model(
    model,
    test_loader,
    out_dim: int,
    out_scaler=None,
    scale_outputs: bool = False,
    print_summary: bool = True,
):
    """
    Evaluate a regression model on a test set.

    Args:
        model:        trained PyTorch model
        test_loader:  DataLoader for the test set
        out_dim:      number of output dimensions
        out_scaler:   optional sklearn-like scaler used on outputs
        scale_outputs: if True, invert scaling using `out_scaler`
        print_summary: if True, print per-output and overall metrics

    Returns:
        y_true:  (N, out_dim) array in original units
        y_pred:  (N, out_dim) array in original units
        mae:     (out_dim,) MAE per output
        rmse:    (out_dim,) RMSE per output
        r2:      (out_dim,) R^2 per output
    """
    model.eval()
    device = next(model.parameters()).device

    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pb = model(xb).cpu().numpy()
            preds.append(pb)
            trues.append(yb.cpu().numpy())

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)

    # Invert scaling if requested and possible
    if scale_outputs and out_scaler is not None:
        y_pred_plot = out_scaler.inverse_transform(y_pred)
        y_true_plot = out_scaler.inverse_transform(y_true)
    else:
        y_pred_plot = y_pred
        y_true_plot = y_true

    # Per-output metrics
    mae = np.array([
        mean_absolute_error(y_true_plot[:, i], y_pred_plot[:, i])
        for i in range(out_dim)
    ])
    rmse = np.array([
        mean_squared_error(y_true_plot[:, i], y_pred_plot[:, i]) ** 0.5
        for i in range(out_dim)
    ])
    r2 = np.array([
        r2_score(y_true_plot[:, i], y_pred_plot[:, i])
        for i in range(out_dim)
    ])

    if print_summary:
        print("=== Test Metrics per Output ===")
        for i in range(out_dim):
            print(f"Output {i}: MAE={mae[i]:.6f}, RMSE={rmse[i]:.6f}, R^2={r2[i]:.4f}")
        print(f"\nOverall MAE:  {mae.mean():.6f}")
        print(f"Overall RMSE: {rmse.mean():.6f}")
        print(f"Mean R^2:     {r2.mean():.4f}")

    return y_true_plot, y_pred_plot, mae, rmse, r2



#### plot the comparison between true and predicted values ####
def plot_parity_plots(
    y_true_plot: np.ndarray,
    y_pred_plot: np.ndarray,
    mae: np.ndarray,
    r2: np.ndarray,
    out_dim: int,
    suptitle: str = "Parity Plots: Ground Truth vs Predicted Parameters",
):
    """
    Create parity plots (true vs predicted) for each output dimension.

    Args:
        y_true_plot: (N, out_dim) array of true values (in original units)
        y_pred_plot: (N, out_dim) array of predicted values (in original units)
        mae:         (out_dim,) MAE per output
        r2:          (out_dim,) R² per output
        out_dim:     number of outputs
        suptitle:    figure title
    """
    # Grid layout: roughly square
    cols = int(math.ceil(math.sqrt(out_dim)))
    rows = int(math.ceil(out_dim / cols))

    plt.figure(figsize=(4 * cols, 3.5 * rows))

    y_min = np.minimum(y_true_plot.min(axis=0), y_pred_plot.min(axis=0))
    y_max = np.maximum(y_true_plot.max(axis=0), y_pred_plot.max(axis=0))

    for i in range(out_dim):
        ax = plt.subplot(rows, cols, i + 1)

        # Scatter: predicted vs true
        ax.scatter(y_true_plot[:, i], y_pred_plot[:, i], s=10, alpha=0.6)

        # Reference line y = x
        lo = min(y_min[i], y_max[i])
        hi = max(y_min[i], y_max[i])
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1)  # black dashed y=x line

        ax.set_xlabel(f"True param{i+1}")
        ax.set_ylabel(f"Pred param{i+1}")
        ax.set_title(f"Output {i+1} — R²={r2[i]:.3f}, MAE={mae[i]:.3g}")

        # Equal aspect improves visual comparison
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")

    plt.suptitle(suptitle, y=1.02, fontsize=12)
    plt.tight_layout()
    plt.show()

