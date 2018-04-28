import numpy as np
from scipy import sparse

def get_lxly(n):
    I = np.arange(n).reshape(int(np.round(np.sqrt(n))), int(np.round(np.sqrt(n))))
    Lx = sparse.diags([1, -1], [0, -1], shape=(n, n)).tolil()
    Ly = sparse.diags([1, -1], [0, -int(np.round(np.sqrt(n)))], shape=(n, n)).tolil()
    Lx[I[:, 0]] = 0
    Ly[I[0]] = 0
    return Lx, Ly

def cgls(A, b, eta=0.0, n_tocare=None, niter=100):
    M,N = A.shape
    if not n_tocare:
        n_tocare = M
    x = np.zeros(N)
    d = b
    r = A.T @ d
    p = r
    y = A @ p

    tau = 1.1
    normr_prev = np.linalg.norm(r) + 1000
    normr = np.linalg.norm(r)
    x_prev = x
    it = 0

    while (np.linalg.norm(r[:n_tocare]) > eta * tau) and (it<niter):
        it = it+1
        alpha = normr**2 / np.linalg.norm(y)**2

        x_prev = x
        x = x + alpha * p
        d = d - alpha * y
        r = A.T @ d
        normr_prev = normr;
        normr = np.linalg.norm(r);
        beta = normr**2 / normr_prev**2;
        p = r + beta * p;
        y = A @ p;
    return x_prev

def del2_matrix(imshape):
    n = np.prod(imshape)
    I = np.arange(n).reshape(*imshape)
    tails = []
    centers = []
    centers.append(I[:, :-1].reshape(-1))
    tails.append(I[:, 1:].reshape(-1))
    centers.append(I[:, 1:].reshape(-1))
    tails.append(I[:, :-1].reshape(-1))

    centers.append(I[1:, :].reshape(-1))
    tails.append(I[:-1, :].reshape(-1))
    centers.append(I[:-1, :].reshape(-1))
    tails.append(I[1:, :].reshape(-1))
    centers = np.concatenate(centers)
    tails = np.concatenate(tails)
    values = np.zeros_like(tails) + 0.25
    L = sparse.coo_matrix((values, (tails, centers))).tocsr()
    L -= sparse.eye(np.prod(im.shape))
    return L

def u_update(A, b, z1, z2, b1, b2, alpha, lda, Dx, Dy, cgls_iter=20):
    lhs = sparse.vstack(((2/(alpha*lda))*A, Dx, Dy))
    rhs = np.concatenate((2*b/(alpha*lda), z1-b1, z2-b2))
    return cgls(lhs, rhs, niter=cgls_iter)
#     return sparse.linalg.lsqr(lhs, rhs, show=True)[0]
def split_bregman(A, b, alpha, lda, niter=100):
    n = A.shape[1]
    i = lambda: np.zeros(n)
    bx = i(); by = i()
    dx = i(); dy = i()
    u = i()
    gx = i()
    gy = i()
    m, = b.shape
    Dx, Dy = get_lxly(n)
    Dxt = Dx.T
    Dyt = Dy.T

    shrink = lambda a,kappa: np.sign(a) * np.maximum(np.abs(a)-kappa, 0);
    it = 0
    while it < niter:
        it += 1
        u = u_update(A, b, gx, gy, bx, by, alpha, lda, Dx, Dy)
        gx = Dx @ u
        gy = Dy @ u
        dx = shrink(gx+bx, 1/lda)
        dy = shrink(gy+by, 1/lda)
        bx = bx + gx - dx
        by = by + gy - dy
#         u = real(u)/scale
    return u

def gamma_reconstruct(A, b, noiselev, theta0, alpha, tol=1e-4, niter=100, verbose=True):
    m, n = A.shape
    Lx, Ly = get_lxly(n)
    thetas = np.ones(n)
    varp = thetas + 100
    it = 0
    while (np.linalg.norm(varp - thetas) / np.linalg.norm(varp) > tol) and (it < niter):
        it = it + 1
        varp = thetas
        dhi = sparse.diags(thetas**(-0.5))
        lhs = sparse.vstack(((1 / noiselev) * A, dhi @ Lx, dhi @ Ly))
        rhs = np.concatenate(((1 / noiselev) * b, np.zeros(Lx.shape[0] + Ly.shape[0])))
        xr = cgls(lhs, rhs, np.sqrt(m), m)
        eta = 0.5*(alpha-2)
        term1 = ((Lx@xr)**2 + (Ly@xr)**2) / (2*theta0)
        thetas = theta0 * (eta + np.sqrt(term1 + eta**2));
        if verbose and (it % 50 == 0):
            print('Iteration %d complete' % it)
    return xr