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

from MLPpackage import MLP, count_params

#### helper to build, train, time, and evaluate one model ####
def run_experiment(hidden, *, in_dim, out_dim, train_loader, val_loader, test_loader,
                   model_cls=MLP, lr=1e-3, wd=1e-5, epochs=120, patience=20, device=None):
    """
    Builds an MLP with 'hidden', trains it, measures wall-clock time,
    and returns metrics, parameter count, and learning curves.
    """
    # Build fresh model
    m = MLP(in_dim=in_dim, hidden=hidden, out_dim=out_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = m.to(device)

    # Optimizer/loss
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)

    # Train with early stopping (simple inline loop)
    best_val = float('inf')
    best_state, wait = None, 0
    train_losses, val_losses = [], []

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        # --- train ---
        m.train()
        tr_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = m(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            tr_sum += loss.item() * xb.size(0)
        tr_loss = tr_sum / len(train_loader.dataset)

        # --- validate ---
        m.eval()
        va_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = m(xb)
                loss = criterion(pred, yb)
                va_sum += loss.item() * xb.size(0)
        va_loss = va_sum / len(val_loader.dataset)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        # early stopping
        if va_loss < best_val - 1e-8:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # Load best weights and measure time
    if best_state is not None:
        m.load_state_dict(best_state)
    elapsed = time.perf_counter() - t0

    # --- Test evaluation (overall MAE/RMSE) in original units if scaler_out exists ---
    m.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            pb = m(xb.to(device)).cpu().numpy()
            preds.append(pb)
            trues.append(yb.cpu().numpy())
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)

    if 'SCALE_OUTPUTS' in globals() and SCALE_OUTPUTS and 'scaler_out' in globals():
        y_pred_eval = scaler_out.inverse_transform(y_pred)
        y_true_eval = scaler_out.inverse_transform(y_true)
    else:
        y_pred_eval = y_pred
        y_true_eval = y_true

    mae = mean_absolute_error(y_true_eval, y_pred_eval, multioutput='raw_values')
    mse = mean_squared_error(y_true_eval, y_pred_eval, multioutput='raw_values')
    rmse = np.sqrt(mean_squared_error(y_true_eval, y_pred_eval, multioutput='raw_values'))
    overall_mae = mae.mean()
    overall_mse = mse.mean()
    overall_rmse = rmse.mean()

    return {
        "model": m,
        "hidden": tuple(hidden),
        "params": count_params(m),
        "best_val_mse": float(best_val),
        "overall_test_mae": float(overall_mae),
        "overall_test_mse": float(overall_mse),
        "overall_test_rmse": float(overall_rmse),
        "time_sec": float(elapsed),
        "train_loss_curve": train_losses,
        "val_loss_curve": val_losses,
    }


#### Plot the comparison of model performances ####
def plot_model_comparison(results, names=None, test_key="overall_test_mse",
                          title="Model Comparison Summary (log-scaled errors)",
                          figsize=(12, 8), rotate=25):
    """
    Plot a 2x2 summary: Val MSE (log), Test metric (log), #Params (log), Time (s).
    
    Args:
        results: dict[name] -> {
            "best_val_mse": float,
            test_key: float (e.g., "overall_test_mse" or "overall_test_rmse"),
            "params": int,
            "time_sec": float,
        }
        names: optional list[str] to set/lock order; defaults to list(results.keys()).
        test_key: key in results for test error (default "overall_test_mse").
        title: figure suptitle.
        figsize: tuple figure size.
        rotate: xtick label rotation degrees.
    """
    if names is None:
        names = list(results.keys())

    val_mse = [results[k]["best_val_mse"] for k in names]
    test_err = [results[k][test_key] for k in names]
    params = [results[k]["params"] for k in names]
    times = [results[k]["time_sec"] for k in names]

    x = np.arange(len(names))
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, y=1.02)

    # (1) Validation MSE (log)
    axes[0, 0].bar(x, val_mse, color=colors[0])
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(names, rotation=rotate, ha="right")
    axes[0, 0].set_ylabel("Best Val MSE (log)")
    axes[0, 0].set_title("Validation Loss")

    # (2) Test error (log)
    axes[0, 1].bar(x, test_err, color=colors[1])
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xticks(x); axes[0, 1].set_xticklabels(names, rotation=rotate, ha="right")
    axes[0, 1].set_ylabel(f"Test {test_key.split('_')[-1].upper()} (log)")
    axes[0, 1].set_title("Test Error")

    # (3) Parameter count (log)
    axes[1, 0].bar(x, params, color=colors[2])
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(names, rotation=rotate, ha="right")
    axes[1, 0].set_ylabel("# Trainable Parameters (log)")
    axes[1, 0].set_title("Model Size")

    # (4) Training time (linear)
    axes[1, 1].bar(x, times, color=colors[3])
    axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(names, rotation=rotate, ha="right")
    axes[1, 1].set_ylabel("Training Time (s)")
    axes[1, 1].set_title("Training Time")

    plt.tight_layout()
    plt.show()


#### plot the validations curves of the models ####
def plot_val_curves(results, names=None, log=True, figsize=(5,4), title="Validation Curves by Architecture"):
    """
    Plot validation-loss curves for multiple runs stored in `results`.

    Args:
        results: dict[name] -> {"val_loss_curve": list/array of floats}
        names: optional list[str] to control plotting order; defaults to results.keys()
        log: if True, use log scale on y-axis
        figsize: tuple for figure size
        title: plot title
    """
    if names is None:
        names = list(results.keys())

    plt.figure(figsize=figsize)
    for name in names:
        curve = results[name]["val_loss_curve"]
        plt.plot(curve, label=name)
    if log:
        plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Val MSE" + (" (log)" if log else ""))
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

    
#### to visualise the split of dataset based on volume fraction ####
def plot_vf_split(vol_sorted, mid, figsize=(6,3)):
    plt.figure(figsize=figsize)
    x = np.arange(len(vol_sorted))
    plt.scatter(x, vol_sorted, s=10, alpha=0.6, label="samples")
    plt.axvline(mid, color="purple", linestyle="--", label="split index")
    plt.xlabel("Sample index (sorted by volume fraction)")
    plt.ylabel("Volume fraction")
    plt.title("Volume fraction of dataset with split point")
    plt.legend()
    plt.tight_layout()
    plt.show()


#### list the original and rotated cell properties to compare ####
def print_property_table(rows, headers=("pattern","D11","D12","D13","D22","D23","D33","vf"),
                         d_decimals=5, vf_decimals=4, title="Property Comparison for Rotated Unit Cells"):
    """
    Print a neatly aligned table of properties without pandas.

    Args:
        rows: iterable of (name, feat) where feat = [D11,D12,D13,D22,D23,D33,vf]
        headers: column names
        d_decimals: decimals for D-components
        vf_decimals: decimals for volume fraction
        title: heading printed above the table
    """
    # format rows
    table_rows = []
    for name, feat in rows:
        vals = [f"{feat[i]:.{d_decimals}f}" for i in range(6)] + [f"{feat[6]:.{vf_decimals}f}"]
        table_rows.append([name] + vals)

    # column widths
    col_w = [max(len(headers[i]), max(len(r[i]) for r in table_rows)) + 2 for i in range(len(headers))]

    # header + separator
    header_row = "".join(headers[i].ljust(col_w[i]) for i in range(len(headers)))
    sep = "-" * len(header_row)

    # print
    print(f"=== {title} ===")
    print(header_row)
    print(sep)
    for r in table_rows:
        print("".join(r[i].ljust(col_w[i]) for i in range(len(headers))))

        
#### plot the comparison between true and predicted property values ####
def plot_property_parity_plots(
    prop_true: np.ndarray,
    prop_pred: np.ndarray,
    mae: np.ndarray,
    r2: np.ndarray,
    out_dim: int,
    prop_names: list,
    suptitle: str = "Parity Plots: True vs Predicted Properties",
):
    """
    Create parity plots (true vs predicted) for each property dimension.

    Args:
        prop_true: (N, out_dim) array of true property values
        prop_pred: (N, out_dim) array of predicted property values
        mae:       (out_dim,) MAE per property
        r2:        (out_dim,) R² per property
        out_dim:   number of property outputs
        prop_names:list of property names (length = out_dim)
        suptitle:  figure title
    """

    cols = int(math.ceil(math.sqrt(out_dim)))
    rows = int(math.ceil(out_dim / cols))

    plt.figure(figsize=(4 * cols, 3.5 * rows))

    vmin = np.minimum(prop_true.min(axis=0), prop_pred.min(axis=0))
    vmax = np.maximum(prop_true.max(axis=0), prop_pred.max(axis=0))

    for i in range(out_dim):
        ax = plt.subplot(rows, cols, i + 1)

        ax.scatter(prop_true[:, i], prop_pred[:, i], s=10, alpha=0.6)

        lo, hi = vmin[i], vmax[i]
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)

        name = prop_names[i]
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(f"{name}  |  R²={r2[i]:.3f}, MAE={mae[i]:.3g}")

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")

    plt.suptitle(suptitle, y=1.02, fontsize=12)
    plt.tight_layout()
    plt.show()


        
#### evaluate inverse generator by property consistency and parameter reconstruction ####
def evaluate_inverse_mapping(
    Inversegen,
    Proppred,
    inv_test_loader,
    device,
    Param_test=None,
    print_report=True
):
    """
    Evaluate inverse mapping on a test set.

    Part 1: Property-consistency
        Compare prop_hat = Proppred(Inversegen(prop_true)) to prop_true.

    Part 2 (optional): Parameter error
        If Param_test is provided, compare param_hat = Inversegen(prop_true) to Param_test.

    Args:
        Inversegen: torch.nn.Module mapping properties -> parameters (B,7)->(B,4)
        Proppred:   frozen torch.nn.Module mapping parameters -> properties (B,4)->(B,7)
        inv_test_loader: DataLoader yielding batches like (prop_batch,)
        device: torch.device
        Param_test: optional torch.Tensor/np.ndarray with shape (N,4)
        print_report: bool, whether to print formatted metrics

    Returns:
        metrics: dict with keys:
            - prop: {mae, rmse, r2, overall}
            - param: {mae, rmse, r2, overall} (only if Param_test provided)
    """
    Inversegen.eval()
    Proppred.eval()

    with torch.no_grad():
        preds_prop, trues_prop = [], []
        for (prop_batch,) in inv_test_loader:
            prop_batch = prop_batch.to(device)
            param_hat = Inversegen(prop_batch)     # (B, 4)
            prop_hat = Proppred(param_hat)         # (B, 7)
            preds_prop.append(prop_hat.detach().cpu().numpy())
            trues_prop.append(prop_batch.detach().cpu().numpy())

    y_pred_prop = np.concatenate(preds_prop, axis=0)
    y_true_prop = np.concatenate(trues_prop, axis=0)

    mae_prop = np.mean(np.abs(y_pred_prop - y_true_prop), axis=0)
    rmse_prop = np.sqrt(np.mean((y_pred_prop - y_true_prop) ** 2, axis=0))
    r2_prop = np.array(
        [r2_score(y_true_prop[:, i], y_pred_prop[:, i]) for i in range(y_true_prop.shape[1])]
    )

    metrics = {
        "prop": {
            "mae": mae_prop,
            "rmse": rmse_prop,
            "r2": r2_prop,
            "overall": {
                "mae": float(mae_prop.mean()),
                "rmse": float(rmse_prop.mean()),
                "mean_r2": float(r2_prop.mean()),
            },
        }
    }

    if print_report:
        print("=== Test property consistency (per output) ===")
        for i in range(y_true_prop.shape[1]):
            print(
                f"d{i+1}: MAE={mae_prop[i]:.6f}  RMSE={rmse_prop[i]:.6f}  R^2={r2_prop[i]:.4f}"
            )
        o = metrics["prop"]["overall"]
        print(f"Overall  MAE={o['mae']:.6f}  RMSE={o['rmse']:.6f}  Mean R^2={o['mean_r2']:.4f}")

    if Param_test is not None:
        if torch.is_tensor(Param_test):
            param_true_test = Param_test.detach().cpu().numpy()
        else:
            param_true_test = np.asarray(Param_test)

        with torch.no_grad():
            pred_params = []
            for (prop_batch,) in inv_test_loader:
                prop_batch = prop_batch.to(device)
                param_hat = Inversegen(prop_batch)
                pred_params.append(param_hat.detach().cpu().numpy())
        param_hat_test = np.concatenate(pred_params, axis=0)

        mae_param = np.mean(np.abs(param_hat_test - param_true_test), axis=0)
        rmse_param = np.sqrt(np.mean((param_hat_test - param_true_test) ** 2, axis=0))
        r2_param = np.array(
            [r2_score(param_true_test[:, i], param_hat_test[:, i]) for i in range(param_true_test.shape[1])]
        )

        metrics["param"] = {
            "mae": mae_param,
            "rmse": rmse_param,
            "r2": r2_param,
            "overall": {
                "mae": float(mae_param.mean()),
                "rmse": float(rmse_param.mean()),
                "mean_r2": float(r2_param.mean()),
            },
        }

        if print_report:
            print("\n=== Test parameter error (per output, sanity check) ===")
            for i in range(param_true_test.shape[1]):
                print(
                    f"r{i+1}: MAE={mae_param[i]:.6f}  RMSE={rmse_param[i]:.6f}  R^2={r2_param[i]:.4f}"
                )
            o = metrics["param"]["overall"]
            print(f"Overall  MAE={o['mae']:.6f}  RMSE={o['rmse']:.6f}  Mean R^2={o['mean_r2']:.4f}")

    return metrics


#### plot the inverse generator output cell property, compared with target ####
def plot_inverse_property_consistency(
    Inversegen,
    Proppred,
    inv_test_loader,
    device,
    scaler_out=None,
    scale_outputs=False,
    prop_names=None,
    figsize_base=(4.0, 3.5),
    point_size=10,
    alpha=0.6,
    show=True
):
    """
    Create parity plots for property consistency:
        True properties vs Proppred(Inversegen(true properties))

    Args:
        Inversegen: torch.nn.Module (properties -> parameters)
        Proppred: torch.nn.Module (parameters -> properties), assumed frozen
        inv_test_loader: DataLoader yielding (prop_batch,)
        device: torch.device
        scaler_out: optional sklearn scaler used on outputs
        scale_outputs: bool, whether to inverse_transform with scaler_out
        prop_names: list of property names (len = output dim)
        figsize_base: base (width, height) per subplot
        point_size: scatter marker size
        alpha: scatter transparency
        show: whether to call plt.show()

    Returns:
        results: dict containing
            - y_true
            - y_pred
            - mae (per output)
            - r2 (per output)
    """
    Inversegen.eval()
    Proppred.eval()

    preds, trues = [], []
    with torch.no_grad():
        for (prop_batch,) in inv_test_loader:
            prop_batch = prop_batch.to(device)
            param_hat = Inversegen(prop_batch)
            prop_hat = Proppred(param_hat)
            preds.append(prop_hat.detach().cpu().numpy())
            trues.append(prop_batch.detach().cpu().numpy())

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)

    if scale_outputs and scaler_out is not None:
        y_pred_plot = scaler_out.inverse_transform(y_pred)
        y_true_plot = scaler_out.inverse_transform(y_true)
    else:
        y_pred_plot = y_pred
        y_true_plot = y_true

    out_dim = y_true_plot.shape[1]

    mae = np.array([
        mean_absolute_error(y_true_plot[:, i], y_pred_plot[:, i])
        for i in range(out_dim)
    ])
    r2 = np.array([
        r2_score(y_true_plot[:, i], y_pred_plot[:, i])
        for i in range(out_dim)
    ])

    if prop_names is None:
        prop_names = [f"d{i+1}" for i in range(out_dim)]

    cols = int(math.ceil(math.sqrt(out_dim)))
    rows = int(math.ceil(out_dim / cols))

    fig_w = figsize_base[0] * cols
    fig_h = figsize_base[1] * rows
    plt.figure(figsize=(fig_w, fig_h))

    for i in range(out_dim):
        ax = plt.subplot(rows, cols, i + 1)

        ax.scatter(
            y_true_plot[:, i],
            y_pred_plot[:, i],
            s=point_size,
            alpha=alpha
        )

        lo = min(y_true_plot[:, i].min(), y_pred_plot[:, i].min())
        hi = max(y_true_plot[:, i].max(), y_pred_plot[:, i].max())
        ax.plot([lo, hi], [lo, hi], "k-", linewidth=1)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel(f"True {prop_names[i]}")
        ax.set_ylabel(f"Pred {prop_names[i]}")
        ax.set_title(f"{prop_names[i]}  R²={r2[i]:.3f}, MAE={mae[i]:.3g}")

    plt.suptitle(
        "Inversegen property consistency (true vs predicted)",
        y=1.02,
        fontsize=13
    )
    plt.tight_layout()

    if show:
        plt.show()

    return {
        "y_true": y_true_plot,
        "y_pred": y_pred_plot,
        "mae": mae,
        "r2": r2,
    }