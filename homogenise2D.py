# Homogenisation 2D
import itertools
import matplotlib.pyplot as plt
import numpy as np

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def compute_lambda_constants(E, nu):
    """
    Compute lambda and mu values for a two-material system:
    - Entry 0 is always the void material (E = 1e-9, nu = 0.3)
    - Entry 1 is computed from the provided E and nu
    Returns:
        lambda_vals : np.array([lambda_void, lambda_material])
        mu_vals     : np.array([mu_void, mu_material])
    """
    
    # ----- Void material -----
    E_void = 1e-9
    nu_void = 0.3
    
    lambda_void = E_void * nu_void / ((1 + nu_void) * (1 - 2 * nu_void))
    mu_void     = E_void / (2 * (1 + nu_void))
    
    # ----- Real material -----
    lambda_mat = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu_mat     = E / (2 * (1 + nu))
    
    # Pack into arrays
    lambda_vals = np.array([lambda_void, lambda_mat])
    mu_vals     = np.array([mu_void, mu_mat])
    
    return lambda_vals, mu_vals


def homogenise(lx, ly, lambda_vals, mu_vals, phi, x):
    """
    Python version of the 2D homogenization.

    Parameters
    ----------
    lx, ly : float
        Unit cell lengths in x and y directions.
    lambda_vals : array_like, shape (2,)
        Lame's first parameter for the two materials.
    mu_vals : array_like, shape (2,)
        Lame's second parameter for the two materials.
    phi : float
        Angle between horizontal and vertical cell wall (degrees).
    x : ndarray, shape (nely, nelx)
        Material indicator matrix (0 or 1).

    Returns
    -------
    CH : ndarray, shape (3, 3)
        Homogenized elasticity tensor.
    """
    x = np.asarray(x)
    x = x+1 # Convert from python index to matlab 
    nely, nelx = x.shape
    nel = nelx * nely

    dx = lx / nelx
    dy = ly / nely

    # Element matrices and vectors
    keLambda, keMu, feLambda, feMu = elementMatVec(dx / 2.0, dy / 2.0, phi)

    # ---------- NODE NUMBERING FOR FULL MESH (1-based) ----------
    nn = (nelx + 1) * (nely + 1)          # total number of nodes
    nnP = nelx * nely                     # number of unique nodes after periodicity
    ndof = 2 * nnP                        # number of dofs

    # nodenrs: (1+nely) x (1+nelx), 1-based numbering
    nodenrs = np.arange(1, (nelx + 1) * (nely + 1) + 1).reshape(
        (nely + 1, nelx + 1), order="F"
    )

    # edofVec: reshape(2*nodenrs(1:end-1,1:end-1)+1, nel, 1)
    edofVec = (2 * nodenrs[:-1, :-1] + 1).reshape(nel, 1, order="F")

    # pattern: [0 1 2*nely+[2 3 0 1] -2 -1]
    pattern = np.array(
        [0, 1, 2 * nely + 2, 2 * nely + 3, 2 * nely + 0, 2 * nely + 1, -2, -1],
        dtype=int,
    )
    edofMat_full = edofVec + pattern  # still 1-based

    # ---------- PERIODIC BOUNDARY CONDITIONS ----------
    # nnPArray: nely x nelx, 1-based
    nnPArray = np.arange(1, nnP + 1).reshape((nely, nelx), order="F")

    # Extend with mirror of top border
    nnP_ext = np.zeros((nely + 1, nelx), dtype=int)
    nnP_ext[:nely, :] = nnPArray
    nnP_ext[nely, :] = nnPArray[0, :]

    # Extend with mirror of left border
    nnP_ext2 = np.zeros((nely + 1, nelx + 1), dtype=int)
    nnP_ext2[:, :nelx] = nnP_ext
    nnP_ext2[:, nelx] = nnP_ext2[:, 0]

    # Dof vector for periodic mapping (1-based dof numbers)
    dofVector = np.zeros(2 * nn, dtype=int)
    temp_nodes = nnP_ext2.reshape(-1, order="F")  # 1-based node numbers
    dofVector[0::2] = 2 * temp_nodes - 1
    dofVector[1::2] = 2 * temp_nodes

    # Map element dofs through periodic dofVector
    # edofMat_full and dofVector are 1-based -> convert index to 0-based for Python
    edofMat_mapped_1based = dofVector[edofMat_full - 1]
    # convert final dof numbers to 0-based indexing
    edofMat = edofMat_mapped_1based - 1  # shape: (nel, 8), 0-based dof indices

    # ---------- MATERIAL PROPERTIES PER ELEMENT ----------
    lambda_field = lambda_vals[0] * (x == 1) + lambda_vals[1] * (x == 2)
    mu_field = mu_vals[0] * (x == 1) + mu_vals[1] * (x == 2)

    # Flatten in column-major order
    lambda_flat = lambda_field.reshape(-1, order="F")
    mu_flat = mu_field.reshape(-1, order="F")

    # ---------- ASSEMBLE GLOBAL STIFFNESS MATRIX (sparse) ----------
    K = lil_matrix((ndof, ndof), dtype=float)

    for e in range(nel):
        lam_e = lambda_flat[e]
        mu_e = mu_flat[e]
        ke_e = lam_e * keLambda + mu_e * keMu  # 8x8
        dofs = edofMat[e, :]
        # add to global K
        for a in range(8):
            ra = dofs[a]
            for b in range(8):
                cb = dofs[b]
                K[ra, cb] += ke_e[a, b]

    # ---------- ASSEMBLE LOAD VECTORS (3 load cases) ----------
    F = np.zeros((ndof, 3), dtype=float)

    for e in range(nel):
        lam_e = lambda_flat[e]
        mu_e = mu_flat[e]
        fe_e = lam_e * feLambda + mu_e * feMu  # 8x3
        dofs = edofMat[e, :]
        F[dofs, :] += fe_e

    # ---------- SOLVE K * chi = F WITH ONE FIXED NODE ----------
    chi = np.zeros((ndof, 3), dtype=float)
    free = np.arange(2, ndof)  # fix dofs 0 and 1 to zero

    # Solve for each load case
    K_free = K[free, :][:, free].tocsr()
    for i_case in range(3):
        chi[free, i_case] = spsolve(K_free, F[free, i_case])

    # ---------- HOMOGENIZATION ----------
    # Element-level displacement vectors for unit strain cases
    nel = nelx * nely
    chi0 = np.zeros((nel, 8, 3), dtype=float)

    # Build chi0_e by solving a reduced element problem
    ke = keMu + keLambda
    fe = feMu + feLambda
    chi0_e = np.zeros((8, 3), dtype=float)

    # Python indices [2, 4, 5, 6, 7]
    idx = np.array([2, 4, 5, 6, 7])
    ke_red = ke[np.ix_(idx, idx)]
    fe_red = fe[idx, :]  # 5x3

    chi0_e[idx, :] = np.linalg.solve(ke_red, fe_red)

    # epsilon0_11 = (1,0,0)
    chi0[:, :, 0] = np.tile(chi0_e[:, 0], (nel, 1))
    # epsilon0_22 = (0,1,0)
    chi0[:, :, 1] = np.tile(chi0_e[:, 1], (nel, 1))
    # epsilon0_12 = (0,0,1)
    chi0[:, :, 2] = np.tile(chi0_e[:, 2], (nel, 1))

    # Flatten chi in column-major order
    chi_vec = chi.reshape(-1, order="F")  # length 3*ndof

    CH = np.zeros((3, 3), dtype=float)
    cellVolume = lx * ly

    # Precompute chi per element per load case: shape (3, nel, 8)
    chi_elem = np.zeros((3, nel, 8), dtype=float)
    for i_case in range(3):
        offset = i_case * ndof
        idx_global = edofMat + offset
        chi_elem[i_case, :, :] = chi_vec[idx_global]

    for i in range(3):
        for j in range(3):
            diff_i = chi0[:, :, i] - chi_elem[i, :, :]  # nel x 8
            diff_j = chi0[:, :, j] - chi_elem[j, :, :]  # nel x 8

            sumLambda_mat = (diff_i @ keLambda) * diff_j
            sumMu_mat = (diff_i @ keMu) * diff_j

            # Sum over local dofs
            sumLambda_el = sumLambda_mat.sum(axis=1)  # length nel
            sumMu_el = sumMu_mat.sum(axis=1)

            # Reshape to (nely, nelx) in column-major order
            sumLambda_grid = sumLambda_el.reshape((nely, nelx), order="F")
            sumMu_grid = sumMu_el.reshape((nely, nelx), order="F")

            CH[i, j] = (1.0 / cellVolume) * np.sum(
                lambda_field * sumLambda_grid + mu_field * sumMu_grid
            )

    return CH


def elementMatVec(a, b, phi):
    """
    elementMatVec(a, b, phi).
    Computes element stiffness matrices keLambda, keMu and load vectors feLambda, feMu.
    """
    # Constitutive matrix contributions
    CMu = np.diag([2.0, 2.0, 1.0])
    CLambda = np.zeros((3, 3))
    CLambda[0:2, 0:2] = 1.0

    # Two Gauss points in both directions
    xx = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    yy = xx.copy()
    ww = np.array([1.0, 1.0])

    keLambda = np.zeros((8, 8), dtype=float)
    keMu = np.zeros((8, 8), dtype=float)
    feLambda = np.zeros((8, 3), dtype=float)
    feMu = np.zeros((8, 3), dtype=float)

    L = np.zeros((3, 4), dtype=float)
    L[0, 0] = 1.0
    L[1, 3] = 1.0
    L[2, 1:3] = 1.0

    tan_phi = np.tan(np.deg2rad(phi))

    for xg, wx in zip(xx, ww):
        for yg, wy in zip(yy, ww):
            # Differentiated shape functions (with respect to local coords)
            dNx = 0.25 * np.array([-(1 - yg), (1 - yg), (1 + yg), -(1 + yg)])
            dNy = 0.25 * np.array([-(1 - xg), -(1 + xg), (1 + xg), (1 - xg)])

            # Node coordinates of the skewed element Jacobian construction
            # x-coordinates:
            x_coords = np.array([
                -a,
                 a,
                 a + 2.0 * b / tan_phi,
                 2.0 * b / tan_phi - a
            ])
            # y-coordinates:
            y_coords = np.array([-b, -b, b, b])

            J = np.vstack((dNx, dNy)) @ np.vstack((x_coords, y_coords)).T

            detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            invJ = (1.0 / detJ) * np.array([[ J[1, 1], -J[0, 1]],
                                            [-J[1, 0],  J[0, 0]]])

            weight = wx * wy * detJ

            # Strain-displacement matrix
            G = np.block([
                [invJ, np.zeros((2, 2))],
                [np.zeros((2, 2)), invJ]
            ])

            dN = np.zeros((4, 8), dtype=float)
            dN[0, 0::2] = dNx
            dN[1, 0::2] = dNy
            dN[2, 1::2] = dNx
            dN[3, 1::2] = dNy

            B = L @ G @ dN  # 3 x 8

            keLambda += weight * (B.T @ CLambda @ B)
            keMu += weight * (B.T @ CMu @ B)

            feLambda += weight * (B.T @ CLambda @ np.diag([1.0, 1.0, 1.0]))
            feMu += weight * (B.T @ CMu @ np.diag([1.0, 1.0, 1.0]))

    return keLambda, keMu, feLambda, feMu



def generate_cross_unit_cell(a, rx, ry, show=True):
    """
    Generate a 2D voxel cross pattern of size (a x a).

    Parameters
    ----------
    a : int
        Side length of the voxel grid.
    rx : float
        Relative width of the vertical bar (fraction of a).
    ry : float
        Relative height of the horizontal bar (fraction of a).
    show : bool
        If True, visualize the voxel pattern.

    Returns
    -------
    arr : (a, a) numpy array
        1 = solid, 0 = void.
    """
    if not (0 <= rx <= 1 and 0 <= ry <= 1):
        raise ValueError("rx and ry must be between 0 and 1")

    # Centered coordinates
    coords = np.arange(a) - (a - 1) / 2.0
    X, Y = np.meshgrid(coords, coords, indexing='xy')

    # Dimensions of cross arms
    half_w = (a * rx) / 2.0  # vertical bar half-width
    half_h = (a * ry) / 2.0  # horizontal bar half-height

    # Bars
    vertical   = np.abs(X) <= half_w
    horizontal = np.abs(Y) <= half_h

    arr = (vertical | horizontal).astype(int)

    # -------------------------
    # Optional visualisation
    # -------------------------
    if show:
        plt.figure(figsize=(2, 2))  # display size (in inches)
        plt.imshow(1 - arr, cmap='gray', origin='lower')
        plt.axis('off')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()

    return arr



def generate_BCC_unitcell(
    a: int,
    r1: float, r2: float, r3: float, r4: float,
    return_bool: bool = False,
    show: bool = True
) -> np.ndarray:
    """
    Generate the BCC unit cell pattern

    Parameters
    ----------
    a : int
        Base size (also used as resolution if `res` is None).
    r1, r2, r3, r4 : float
        Ratios for a1..a4 such that a1=r1*a, etc.
    return_bool : bool, optional
        If True, return boolean mask; otherwise return {0,1} uint8.
    show : bool, optional
        If True, display the image.

    Returns
    -------
    np.ndarray
        Pattern as a boolean array (if `return_bool=True`) or uint8 (0/1).
    """
    # --- set up ---
    a = float(a)
    a1, a2, a3, a4 = (r1*a, r2*a, r3*a, r4*a)

    # pixel-center coordinates (i-0.5, j-0.5)
    x = np.arange(a, dtype=float) + 0.5
    y = np.arange(a, dtype=float) + 0.5
    X, Y = np.meshgrid(x, y)  # X ~ xc, Y ~ yc

    # --- helper: y-value of line through (x1,y1)-(x2,y2) at X ---
    # adds tiny epsilon to avoid divide-by-zero in extreme cases
    def line_y(x1, y1, x2, y2, X):
        eps = 1e-12
        return y1 + (X - x1) * (y2 - y1) / ( (x2 - x1) + eps )

    # lines 
    L1y = line_y(0,   a - a1, a - a3, 0,   X)
    L2y = line_y(a1,  a,      a,      a3,  X)
    L3y = line_y(a4,  0,      a,      a - a2, X)
    L4y = line_y(0,   a4,     a - a2, a,   X)

    # "upper is positive" => (yc - y_line) >= 0
    l1 = (Y - L1y) >= 0
    l2 = (Y - L2y) >= 0
    l3 = (Y - L3y) >= 0
    l4 = (Y - L4y) >= 0

    mask = (l1 & (~l2)) | (l3 & (~l4))

    pattern = mask if return_bool else mask.astype(np.uint8)

    if show:
        plt.figure(figsize=(2,2)) # controls the display size (in inches)
        plt.imshow(1-pattern, cmap='gray', origin='lower')
        plt.axis('off')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()

    return pattern


def plot_directional_stiffness(D):
    """
    Visualise the directional stiffness given a 2D elasticity tensor
    """
    S = np.linalg.inv(D)

    thetas = np.linspace(0, 2*np.pi, 360)
    E_theta = []

    for th in thetas:
        c = np.cos(th)
        s = np.sin(th)
        invE = (S[0,0]*c**4 +
                S[1,1]*s**4 +
                (2*S[0,1] + S[2,2])*s**2*c**2)
        E_theta.append(1/invE)

    E_theta = np.array(E_theta)

    plt.figure(figsize=(2,2))
    ax = plt.subplot(111, polar=True)
    ax.plot(thetas, E_theta,linewidth=2.5)
    ax.fill(thetas, E_theta, alpha=0.3) 
    ax.set_title("Directional Young's Modulus")
    ax.grid(True)

    plt.show()