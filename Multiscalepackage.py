"""
TopOpt 165-line style implementation (Aage, Johansen) with a clean stress post-process.

This file provides:
  - topopt2D(...): runs topology optimisation and returns a state dict
  - compute_von_mises_stress(...): computes per-element stresses and von Mises (normalised)
  - plot_stress_masked(...): helper for plotting masked stress

Notes:
  - Stress recovery uses a constant B matrix for a bilinear Q4 element at the element center,
    matching your snippet.
  - The FE model uses the classic 88/165-line cantilever setup.
"""

from __future__ import division

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from sklearn.metrics import mean_absolute_error, r2_score
import torch
from torch import nn


def lk(E=1.0, nu=0.3):
    """Element stiffness matrix for the standard 4-node quad (plane stress), as in the 88/165-line code."""
    k = np.array(
        [
            1 / 2 - nu / 6,
            1 / 8 + nu / 8,
            -1 / 4 - nu / 12,
            -1 / 8 + 3 * nu / 8,
            -1 / 4 + nu / 12,
            -1 / 8 - nu / 8,
            nu / 6,
            1 / 8 - 3 * nu / 8,
        ]
    )
    KE = E / (1 - nu**2) * np.array(
        [
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
        ]
    )
    return KE


def oc(nelx, nely, x, volfrac, dc, dv, g):
    """Optimality criteria update (Nguyen/Paulino style)."""
    l1 = 0.0
    l2 = 1e9
    move = 0.2
    xnew = np.zeros(nelx * nely)

    while (l2 - l1) / (l1 + l2) > 1e-3:
        lmid = 0.5 * (l2 + l1)
        xnew[:] = np.maximum(
            0.0,
            np.maximum(
                x - move,
                np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / dv / lmid))),
            ),
        )
        gt = g + np.sum(dv * (xnew - x))
        if gt > 0:
            l1 = lmid
        else:
            l2 = lmid

    return xnew, gt


def topopt2D(
    nelx,
    nely,
    volfrac,
    penal,
    rmin,
    ft,
    Emin=1e-9,
    Emax=1.0,
    nu=0.3,
    max_iter=2000,
    tol_change=0.01,
    plot_density=True,
):
    """
    Runs minimum compliance topology optimisation with OC.

    Returns a state dict containing at least:
        u, edofMat, nelx, nely, xPhys, nu, Emax, Emin, KE
    """
    print("Minimum compliance problem with OC")
    print("ndes: " + str(nelx) + " x " + str(nely))
    print("volfrac: " + str(volfrac) + ", rmin: " + str(rmin) + ", penal: " + str(penal))
    print("Filter method: " + ["Sensitivity based", "Density based"][ft])

    ndof = 2 * (nelx + 1) * (nely + 1)

    x = volfrac * np.ones(nely * nelx, dtype=float)
    xold = x.copy()
    xPhys = x.copy()

    g = 0.0
    dc = np.zeros(nely * nelx, dtype=float)
    dv = np.ones(nely * nelx, dtype=float)
    ce = np.ones(nely * nelx, dtype=float)

    KE = lk(E=1.0, nu=nu)

    edofMat = np.zeros((nelx * nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx * nely
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edofMat[el, :] = np.array(
                [2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3, 2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1]
            )

    iK = np.kron(edofMat, np.ones((8, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 8))).flatten()

    nfilter = int(nelx * nely * ((2 * (np.ceil(rmin) - 1) + 1) ** 2))
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
    cc = 0
    for i in range(nelx):
        for j in range(nely):
            row = i * nely + j
            kk1 = int(np.maximum(i - (np.ceil(rmin) - 1), 0))
            kk2 = int(np.minimum(i + np.ceil(rmin), nelx))
            ll1 = int(np.maximum(j - (np.ceil(rmin) - 1), 0))
            ll2 = int(np.minimum(j + np.ceil(rmin), nely))
            for k in range(kk1, kk2):
                for l in range(ll1, ll2):
                    col = k * nely + l
                    fac = rmin - np.sqrt((i - k) * (i - k) + (j - l) * (j - l))
                    iH[cc] = row
                    jH[cc] = col
                    sH[cc] = np.maximum(0.0, fac)
                    cc += 1

    H = coo_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).tocsc()
    Hs = H.sum(1)

    dofs = np.arange(ndof)
    fixed = np.union1d(dofs[0 : 2 * (nely + 1) : 2], np.array([ndof - 1]))
    free = np.setdiff1d(dofs, fixed)

    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))
    f[1, 0] = -1.0

    fig = None
    ax = None
    im = None
    if plot_density:
        plt.ion()
        fig, ax = plt.subplots()
        im = ax.imshow(
            -xPhys.reshape((nelx, nely)).T,
            cmap="gray",
            interpolation="none",
            norm=colors.Normalize(vmin=-1, vmax=0),
        )
        #fig.show()

    loop = 0
    change = 1.0

    while change > tol_change and loop < max_iter:
        loop += 1

        sK = (KE.flatten()[:, None] * (Emin + (xPhys**penal) * (Emax - Emin))).flatten(order="F")
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K = K[free, :][:, free]

        u[free, 0] = spsolve(K, f[free, 0])

        ue = u[edofMat].reshape(nelx * nely, 8)
        ce[:] = (ue @ KE * ue).sum(1)
        obj = ((Emin + xPhys**penal * (Emax - Emin)) * ce).sum()
        dc[:] = (-penal * xPhys ** (penal - 1) * (Emax - Emin)) * ce
        dv[:] = 1.0

        if ft == 0:
            dc[:] = (np.asarray((H @ (x * dc)) / Hs).ravel()) / np.maximum(0.001, x)
        elif ft == 1:
            dc[:] = np.asarray(H @ (dc[:, None] / Hs)).ravel()
            dv[:] = np.asarray(H @ (dv[:, None] / Hs)).ravel()

        xold[:] = x
        x[:], g = oc(nelx, nely, x, volfrac, dc, dv, g)

        if ft == 0:
            xPhys[:] = x
        elif ft == 1:
            xPhys[:] = np.asarray(H @ x[:, None] / Hs).ravel()

        change = np.linalg.norm(x.reshape(-1, 1) - xold.reshape(-1, 1), np.inf)

        if plot_density:
            im.set_array(-xPhys.reshape((nelx, nely)).T)
            fig.canvas.draw()

        print(
            "it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(
                loop, obj, (g + volfrac * nelx * nely) / (nelx * nely), change
            )
        )

    if plot_density:
        plt.ioff()
        plt.show()

    state = {
        "u": u,
        "edofMat": edofMat,
        "nelx": nelx,
        "nely": nely,
        "xPhys": xPhys,
        "nu": nu,
        "Emax": Emax,
        "Emin": Emin,
        "penal": penal,
        "KE": KE,
        "free_dofs": free,
        "fixed_dofs": fixed,
        "f": f,
    }
    return state


def compute_element_stress_and_von_mises(
    state,
    sigma_yield,
    density_threshold=0.5,
    mask=True,
    return_grids=True,
):
    """
    Compute per-element stresses (sigma_x, sigma_y, tau_xy) and normalised von Mises stress.

    Args:
        state: dict, must contain keys: u, edofMat, nelx, nely, nu, Emax, xPhys
        sigma_yield: scalar yield stress used for normalisation
        density_threshold: elements below this density can be masked (set to np.nan) for plotting
        mask: if True, also return masked grids for plotting
        return_grids: if True, return grid-shaped arrays (nely, nelx)

    Returns:
        out: dict with keys:
            sigma_elem: (nel, 3) array [sigma_x, sigma_y, tau_xy]
            sigma_x: (nel,)
            sigma_y: (nel,)
            tau_xy: (nel,)
            sigma_vm: (nel,) normalised von Mises

        If return_grids:
            sigma_x_grid, sigma_y_grid, tau_xy_grid, sigma_vm_grid: (nely, nelx)
            density_grid: (nely, nelx)

        If mask and return_grids:
            sigma_x_masked, sigma_y_masked, tau_xy_masked, sigma_vm_masked: (nely, nelx)
    """
    U = state["u"][:, 0]
    edofMat = state["edofMat"]
    nu = float(state["nu"])
    Emax = float(state["Emax"])
    nelx = int(state["nelx"])
    nely = int(state["nely"])

    # Plane stress constitutive matrix
    D = (Emax / (1 - nu**2)) * np.array(
        [
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1 - nu) / 2.0],
        ],
        dtype=float,
    )

    # Strain-displacement matrix at element center (as in your snippet)
    B = 0.25 * np.array(
        [
            [-1, 0, 1, 0, 1, 0, -1, 0],
            [0, -1, 0, -1, 0, 1, 0, 1],
            [-1, -1, -1, 1, 1, 1, 1, -1],
        ],
        dtype=float,
    )

    nel = nelx * nely
    sigma_elem = np.zeros((nel, 3), dtype=float)
    sigma_vm = np.zeros(nel, dtype=float)

    inv_yield = 1.0 / float(sigma_yield)

    for el in range(nel):
        u_e = U[edofMat[el]]          # (8,)
        eps = B @ u_e                 # (3,)
        sig = D @ eps                 # (3,) -> [sx, sy, txy]
        sigma_elem[el, :] = sig

        sx, sy, txy = sig
        vm = math.sqrt(sx * sx - sx * sy + sy * sy + 3.0 * txy * txy)
        sigma_vm[el] = inv_yield * vm

    sigma_x = sigma_elem[:, 0].copy()
    sigma_y = sigma_elem[:, 1].copy()
    tau_xy = sigma_elem[:, 2].copy()

    out = {
        "sigma_elem": sigma_elem,
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
        "tau_xy": tau_xy,
        "sigma_vm": sigma_vm,
    }

    if not return_grids:
        return out

    def to_grid(v):
        # match your existing orientation (flip after reshape-transpose)
        return np.flip(v.reshape((nelx, nely)).T, axis=0)

    sigma_x_grid = to_grid(sigma_x)
    sigma_y_grid = to_grid(sigma_y)
    tau_xy_grid = to_grid(tau_xy)
    sigma_vm_grid = to_grid(sigma_vm)

    density_grid = state["xPhys"].reshape((nelx, nely)).T

    out.update(
        {
            "sigma_x": sigma_x_grid,
            "sigma_y": sigma_y_grid,
            "tau_xy": tau_xy_grid,
            "sigma_vm": sigma_vm_grid,
            "density": density_grid,
        }
    )

    if mask:
        # keep your masking convention
        mask_grid = density_grid >= density_threshold

        out.update(
            {
                "sigma_x_masked": np.flip(np.where(mask_grid, sigma_x_grid, np.nan), axis=0),
                "sigma_y_masked": np.flip(np.where(mask_grid, sigma_y_grid, np.nan), axis=0),
                "tau_xy_masked": np.flip(np.where(mask_grid, tau_xy_grid, np.nan), axis=0),
                "sigma_vm_masked": np.flip(np.where(mask_grid, sigma_vm_grid, np.nan), axis=0),
            }
        )

    return out


def plot_stress_masked(stress_masked, vmin=0.0, vmax=1.0, title=None):
    """Simple plot helper for masked stress grids."""
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(
        stress_masked,
        cmap="coolwarm",
        interpolation="none",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.1)
    
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt


def plot_stress_all(stress, cmap="coolwarm"):
    """
    Plot sigma_x, sigma_y, tau_xy, and von Mises stress in a 2x2 grid.

    Args:
        stress: dict returned by compute_element_stress_and_von_mises
        cmap: matplotlib colormap
    """
    fields = [
        ("sigma_x", r"$\sigma_x$"),
        ("sigma_y", r"$\sigma_y$"),
        ("tau_xy",  r"$\tau_{xy}$"),
        ("sigma_vm", r"$\sigma_{\mathrm{vm}}$"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes = axes.flatten()

    for ax, (key, title) in zip(axes, fields):
        field = stress[key]
        im = ax.imshow(
            field,
            cmap=cmap,
            interpolation="none",
            origin="lower",
            vmin=field.min(),
            vmax=field.max(),
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

#### This function is to plot the optimal De field by picking up the Dij ####
def plot_De_field(De, nelx, nely, idx, cmap="coolwarm"):
    field = np.flip(
        De[:, idx].reshape((nelx, nely)).T,
        axis=0
    )

    plt.figure(figsize=(4, 3))
    im = plt.imshow(
        field,
        cmap=cmap,
        interpolation="none",
        origin="lower",
    )
    plt.colorbar(im, fraction=0.03, pad=0.05)
    De_names = ['D11', 'D12', 'D13', 'D22', 'D23', 'D33', 'Vf']
    plt.title(De_names[idx])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()


#### identify element number based on row and column ####
def pick_De_i(De, ix, iy, nelx, nely):
    """
    Get the De row for element at (ix, iy) using topopt element ordering:
        el = iy + ix*nely
    """
    if ix < 0 or ix >= nelx or iy < 0 or iy >= nely:
        raise IndexError(f"(ix, iy)=({ix}, {iy}) out of bounds for nelx={nelx}, nely={nely}")

    el = iy + ix * nely
    return De[el, :], el


#### predict cell parameter based on elemental D and vf ####
def predict_r_from_De(
    De_row,
    model,
    one_value=1.0,
    zero_value=0.0,
    tol=1e-8,
    device=None,
):
    """
    Torch-safe version.
    """
    last = float(De_row[-1])

    if abs(last - one_value) <= tol:
        return np.array([0.5, 0.5, 0.5, 0.5], dtype=float)

    if abs(last - zero_value) <= tol:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=float)

    if device is None:
        device = next(model.parameters()).device

    # NumPy -> Torch
    x = torch.tensor(
        De_row,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)   # shape (1, 7)

    model.eval()
    with torch.no_grad():
        y = model(x)  # shape (1, 4)

    y = y.detach().cpu().numpy().reshape(-1)

    if y.size != 4:
        raise ValueError(f"Model output must be 4D, got shape {y.shape}")

    return y