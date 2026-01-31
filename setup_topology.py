from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from matplotlib import colors
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

def setup_topology(nelx, nely, rmin, Emin, Emax, nu, penal, volfrac):
    # element stiffness (8x8) -- keep your k/KE definition
    k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
    KE = 1/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
    [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
    [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
    [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
    [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
    [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
    [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
    [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ]);

    ndof = 2*(nelx+1)*(nely+1)

    edofMat=np.zeros((nelx*nely,8),dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely+elx*nely # element number
            n1=(nely+1)*elx+ely
            n2=(nely+1)*(elx+1)+ely
            edofMat[el,:]=np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1])
    
    # Construct the index pointers for the coo format
    iK = np.kron(edofMat,np.ones((8,1))).flatten()
    jK = np.kron(edofMat,np.ones((1,8))).flatten() 

    # BC's and support
    dofs=np.arange(2*(nelx+1)*(nely+1))
    fixed=np.union1d(dofs[0:2*(nely+1):2],np.array([2*(nelx+1)*(nely+1)-1]))
    free=np.setdiff1d(dofs,fixed)
    
    # Solution and RHS vectors
    f=np.zeros((ndof,1))
    u=np.zeros((ndof,1))
        
    # Set load
    f[1,0]=-1

    # Filter: Build (and assemble) the index+data vectors for the coo matrix format
    nfilter=int(nelx*nely*((2*(np.ceil(rmin)-1)+1)**2))
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
    cc=0
    for i in range(nelx):
        for j in range(nely):
            row=i*nely+j
            kk1=int(np.maximum(i-(np.ceil(rmin)-1),0))
            kk2=int(np.minimum(i+np.ceil(rmin),nelx))
            ll1=int(np.maximum(j-(np.ceil(rmin)-1),0))
            ll2=int(np.minimum(j+np.ceil(rmin),nely))
            for k in range(kk1,kk2):
                for l in range(ll1,ll2):
                    col=k*nely+l
                    fac=rmin-np.sqrt(((i-k)*(i-k)+(j-l)*(j-l)))
                    iH[cc]=row
                    jH[cc]=col
                    sH[cc]=np.maximum(0.0,fac)
                    cc=cc+1
                    
    # Finalize assembly and convert to csc format
    H=coo_matrix((sH,(iH,jH)),shape=(nelx*nely,nelx*nely)).tocsc()
    Hs=H.sum(1)

    # Allocate design variables (as array), initialise and allocate sens.
    x=volfrac * np.ones(nely*nelx,dtype=float)
    xold=x.copy()
    xPhys=x.copy()
    g=0 # must be initialised to use the NGuyen/Paulino OC approach
    dc=np.zeros((nely,nelx), dtype=float)

    # bookkeeping arrays
    ce = np.zeros(nelx*nely, dtype=float)
    dc = np.zeros(nelx*nely, dtype=float)
    dv = np.ones(nelx*nely, dtype=float)

    # pack everything into state dict
    state = {
        'nelx': nelx, 'nely': nely, 'rmin': rmin,
        'Emin': Emin, 'Emax': Emax, 'nu': nu, 'penal': penal,
        'volfrac': volfrac,
        'ndof': ndof, 'KE': KE, 'edofMat': edofMat,
        'iK': iK, 'jK': jK,
        'fixed': fixed, 'free': free,
        'f': f, 'u': u,
        'H': H, 'Hs': Hs,
        'x': x, 'xold': xold, 'xPhys': xPhys,
        'ce': ce, 'dc': dc, 'dv': dv,}
    return state

def setup_topology_stress(nelx, nely, rmin, Emin, Emax, nu, penal, volfrac, sigma_limit, p_norm):
    """
    Initialise topology optimisation state for stress-constrained design.
    Stress is normalised by sigma_yield=1.0.
    Constraint limit is sigma_limit.
    """
    # Element stiffness (plane stress 8x8)
    k = np.array([1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8,
                  -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8])
    KE = 1/(1 - nu**2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])

    ndof = 2*(nelx+1)*(nely+1)

    # Element DOF mapping
    edofMat = np.zeros((nelx*nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx*nely
            n1 = (nely+1)*elx + ely
            n2 = (nely+1)*(elx+1) + ely
            edofMat[el, :] = np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,
                                       2*n2, 2*n2+1, 2*n1, 2*n1+1])

    # Sparse stiffness indices
    iK = np.kron(edofMat, np.ones((8,1))).flatten()
    jK = np.kron(edofMat, np.ones((1,8))).flatten()

    # Boundary conditions
    dofs = np.arange(ndof)
    fixed = np.union1d(dofs[0:2*(nely+1):2], np.array([2*(nelx+1)*(nely+1)-1]))
    free = np.setdiff1d(dofs, fixed)

    # Load and displacement vectors
    f = np.zeros((ndof,1))
    u = np.zeros((ndof,1))
    f[1,0] = -1

    # Sensitivity filter
    nfilter = int(nelx*nely*((2*(np.ceil(rmin)-1)+1)**2))
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
    cc = 0
    for i in range(nelx):
        for j in range(nely):
            row = i*nely + j
            kk1 = int(max(i-(np.ceil(rmin)-1),0))
            kk2 = int(min(i+np.ceil(rmin),nelx))
            ll1 = int(max(j-(np.ceil(rmin)-1),0))
            ll2 = int(min(j+np.ceil(rmin),nely))
            for k in range(kk1,kk2):
                for l in range(ll1,ll2):
                    col = k*nely + l
                    fac = rmin - np.sqrt((i-k)**2 + (j-l)**2)
                    iH[cc] = row
                    jH[cc] = col
                    sH[cc] = max(0.0, fac)
                    cc += 1
    H = coo_matrix((sH, (iH, jH)), shape=(nelx*nely, nelx*nely)).tocsc()
    Hs = H.sum(1)

    # Design variables
    x = volfrac*np.ones(nelx*nely)
    xold = x.copy()
    xPhys = x.copy()

    ce = np.zeros(nelx*nely)
    dc = np.zeros(nelx*nely)
    dv = np.ones(nelx*nely)

    state = {
        'nelx': nelx, 'nely': nely, 'rmin': rmin,
        'Emin': Emin, 'Emax': Emax, 'nu': nu, 'penal': penal,
        'volfrac': volfrac,
        'ndof': ndof, 'KE': KE, 'edofMat': edofMat,
        'iK': iK, 'jK': jK,
        'fixed': fixed, 'free': free,
        'f': f, 'u': u,
        'H': H, 'Hs': Hs,
        'x': x, 'xold': xold, 'xPhys': xPhys,
        'ce': ce, 'dc': dc, 'dv': dv,
        # stress constraint
        'sigma_vm': np.zeros(nelx*nely),
        'p_stress': 0.0,
        'c2': 0.0,
        'dc2': np.zeros(nelx*nely),
        'sigma_yield': 1.0,      # normalisation for von Mises
        'sigma_limit': sigma_limit, # 0.3 or desired constraint
        'p_norm': p_norm,
        'lambda_s': 0.0,
        'mu_s': 10.0,
        'max_mu': 1e6,}
    return state
