from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from matplotlib import colors
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

def iterate_topology(state, params):
    nelx = state['nelx']; nely = state['nely']
    KE = state['KE']; edofMat = state['edofMat']
    iK = state['iK']; jK = state['jK']
    free = state['free']; f = state['f']; u = state['u']
    H = state['H']; Hs = state['Hs']
    x = state['x']; xPhys = state['xPhys']
    Emin = params.get('Emin', state['Emin'])
    Emax = params.get('Emax', state['Emax'])
    penal = params.get('penal', state['penal'])
    move = params.get('move', 0.2)
    ndof = 2*(nelx+1)*(nely+1)

    # 1) Assemble global stiffness K (sK vector) and solve
    # sK: element stiffness entries repeated by element
    # Setup and solve FE problem
    sK = ((KE.flatten()[np.newaxis]).T*(Emin+(xPhys)**penal*(Emax-Emin))).flatten(order='F')
    K = coo_matrix((sK,(iK,jK)),shape=(ndof,ndof)).tocsc()

    # sK = (KE.flatten(order='F')[:, np.newaxis] * (Emin + xPhys**penal * (Emax - Emin))).flatten(order='F')
    # # build sparse K and solve (use same approach as your code)
    # K = coo_matrix((sK, (iK, jK)), shape=(state['ndof'], state['ndof'])).tocsc()
    K_reduced = K[state['free'], :][:, state['free']]
    u[state['free'], 0] = spsolve(K_reduced, f[state['free'], 0])

    # 2) Element strain energy and objective
    # u[edofMat] gives (nelx*nely,8,1) style array via fancy indexing; replicate your approach:
    ue = u[edofMat].reshape(nelx*nely, 8)
    ce = (np.dot(ue, KE) * ue).sum(axis=1)   # scalar per element
    obj = ((Emin + xPhys**penal * (Emax - Emin)) * ce).sum()

    # 3) Sensitivities
    dc = (-penal * xPhys**(penal-1) * (Emax - Emin)) * ce
    dv = np.ones_like(dc)

    # 4) Filter sensitivities
    dc = np.asarray(H * (dc[np.newaxis].T / Hs))[:, 0]
    dv = np.asarray(H * (dv[np.newaxis].T / Hs))[:, 0]

    # 5) OC update
    l1 = 0.0; l2 = 1e9; eps = 1e-9
    xnew = np.zeros_like(x)
    while (l2 - l1) / (l1 + l2 + eps) > 1e-3:
        lmid = 0.5*(l2 + l1)
        x_candidate = np.maximum(0.0, np.maximum(x - move,
                             np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / dv / lmid)))))
        if (x_candidate.sum() - state['volfrac']*nelx*nely) > 0:
            l1 = lmid
        else:
            l2 = lmid
    xnew[:] = np.maximum(0.0, np.maximum(x - move,
                     np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / dv / lmid)))))

    # 6) Update state and filtered physical densities
    state['xold'] = x.copy()
    state['x'] = xnew
    arr = np.asarray(H * (xnew[np.newaxis].T / Hs))[:, 0]
    state['xPhys'] = np.round(arr, 2)
    
    # state['xPhys'] = np.round(arr.reshape((nelx,nely)).T,2)
    state['ce'] = ce
    state['dc'] = dc
    state['dv'] = dv

    change = np.max(np.abs(state['x'] - state['xold']))
    info = {'obj': float(obj), 'change': float(change)}
    return state, info

def iterate_topology_stress(state, params):
    """
    Topology iteration with stress constraint using p-norm and Augmented Lagrangian.
    Stress normalised by sigma_yield, constraint = sigma_limit.
    """

    # --- Unpack ---
    nelx = state['nelx']; nely = state['nely']
    KE = state['KE']; edofMat = state['edofMat']
    iK = state['iK']; jK = state['jK']
    free = state['free']; f = state['f']; u = state['u']
    H = state['H']; Hs = state['Hs']
    x = state['x']; xPhys = state['xPhys']
    Emin = params.get('Emin', state['Emin'])
    Emax = params.get('Emax', state['Emax'])
    penal = params.get('penal', state['penal'])
    ndof = state['ndof']
    nu = state['nu']
    sigma_yield = state['sigma_yield']   # normalization
    sigma_limit = state['sigma_limit']   # constraint
    p_norm = state['p_norm']

    # --- FE solve ---
    sK = ((KE.flatten()[np.newaxis]).T * (Emin + xPhys**penal*(Emax-Emin))).flatten(order='F')
    K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
    K_reduced = K[free, :][:, free]
    u[free, 0] = spsolve(K_reduced, f[free, 0])

    # --- Element strain energy & objective ---
    ue = u[edofMat].reshape(nelx*nely, 8)
    ce = (np.dot(ue, KE) * ue).sum(axis=1)
    obj = ((Emin + xPhys**penal*(Emax - Emin)) * ce).sum()

    # --- Sensitivity of objective ---
    dc = (-penal * xPhys**(penal-1) * (Emax - Emin)) * ce
    dv = np.ones_like(dc)

    # --- Filter ---
    dc = np.asarray(H * (dc[np.newaxis].T / Hs))[:,0]
    dv = np.asarray(H * (dv[np.newaxis].T / Hs))[:,0]

    # --- Compute stress & p-norm ---
    def stress_and_pnorm(state, eps_vm=1e-12, eps_S=1e-12):
        xPhys = state['xPhys']
        u = state['u'][:,0]
        nel = nelx*nely

        # Constitutive matrix
        D_unit = (1.0/(1-nu**2)) * np.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
        B_center = 0.25*np.array([[-1,0,1,0,1,0,-1,0],
                                  [0,-1,0,-1,0,1,0,1],
                                  [-1,-1,-1,1,1,1,1,-1]])

        sigma_elem = np.zeros((nel,3))
        sigma_vm = np.zeros(nel)
        eps_elem = np.zeros((nel,3))

        for e in range(nel):
            u_e = u[edofMat[e]]
            eps_e = B_center @ u_e
            eps_elem[e,:] = eps_e
            E_e = Emin + xPhys[e]**penal*(Emax-Emin)
            sig_e = E_e * D_unit @ eps_e
            sigma_elem[e,:] = sig_e
            sx, sy, txy = sig_e
            sigma_vm[e] = np.sqrt(sx**2 - sx*sy + sy**2 + 3*txy**2) / sigma_yield

        vm_safe = np.maximum(sigma_vm, eps_vm)
        S = np.sum(vm_safe**p_norm)/nel + eps_S
        p_stress = S**(1.0/p_norm)
        c2 = p_stress - sigma_limit

        # Sensitivity
        dE_dx = penal * xPhys**(penal-1)*(Emax-Emin)
        prefactor = (1.0/nel) * S**(1.0/p_norm - 1.0)
        dc2 = np.zeros(nel)
        for e in range(nel):
            dsig_dE = D_unit @ eps_elem[e]
            dsig_dx = dsig_dE * dE_dx[e]
            sx, sy, txy = sigma_elem[e] * sigma_yield  # scale back
            vm = vm_safe[e]
            # normalized sensitivity
            dvm_dsig = np.array([(2*sx - sy)/(2*vm*sigma_yield),
                                 (2*sy - sx)/(2*vm*sigma_yield),
                                 3*txy/(vm*sigma_yield)])
            dvm_dx = dvm_dsig @ dsig_dx
            dc2[e] = prefactor * (vm**(p_norm-1)) * dvm_dx

        dc2 = np.asarray(H * (dc2[np.newaxis].T / Hs))[:,0]
        return sigma_vm, p_stress, c2, dc2

    sigma_vm, p_stress, c2, dc2 = stress_and_pnorm(state)

    # --- Augmented Lagrangian ---
    lambda_s = state['lambda_s']
    mu_s = state['mu_s']

    # Dampen AL updates to reduce oscillation
    alpha = 0.5  # damping factor for multiplier effect
    dc_total = dc + alpha * (lambda_s + mu_s*c2) * dc2

    # --- OC update ---
    def oc_update(nelx, nely, x, volfrac, dc, dv):
        l1, l2 = 0.0, 1e9
        move = 0.2
        xnew = np.zeros_like(x)
        eps = 1e-9
        for _ in range(80):
            lmid = 0.5*(l1+l2)
            with np.errstate(divide='ignore', invalid='ignore'):
                arg = np.maximum(1e-9, -dc/dv / lmid)
                x_candidate = np.maximum(0.0, np.maximum(x-move,
                                     np.minimum(1.0, np.minimum(x+move, x*np.sqrt(arg)))))
            if np.sum(x_candidate) - volfrac*nelx*nely > 0:
                l1 = lmid
            else:
                l2 = lmid
            xnew[:] = x_candidate
            if abs(l2-l1)/(l1+l2+eps) < 1e-3:
                break
        return xnew

    xnew = oc_update(nelx, nely, x, state['volfrac'], dc_total, dv)

    # --- Update state ---
    state['xold'] = x.copy()
    state['x'] = xnew
    state['xPhys'] = np.asarray(H * (xnew[np.newaxis].T / Hs))[:,0]
    state['ce'] = ce
    state['dc'] = dc
    state['dv'] = dv
    state['sigma_vm'] = sigma_vm
    state['p_stress'] = p_stress
    state['c2'] = c2
    state['dc2'] = dc2

    # --- Update AL multipliers ---
    state['lambda_s'] = max(0.0, lambda_s + 0.3 * mu_s * c2)  # smaller multiplier step
    if c2 > 0.05:
        state['mu_s'] = min(state['mu_s']*1.05, state['max_mu'])  # slower mu increase

    change = np.max(np.abs(state['x'] - state['xold']))
    info = {'obj': float(obj), 'change': float(change),
            'p_stress': float(p_stress), 'c2': float(c2)}

    return state, info
