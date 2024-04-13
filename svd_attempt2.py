# Based on: https://gist.github.com/thomvolker/2e98eaa778397877da7715f35398d742

import numpy as np


def svd_2(A, u, v, s, n_dim, n_iter, atol):
    X_res = A.copy()
    warn = False

    for d in range(n_dim):
        if d == 1:
            X_res = A - np.dot(u[:, :d], np.dot(np.diag(s[:d]), v[:, :d].T))
        elif d > 1:
            X_res = A - np.dot(u[:, :d], np.dot(np.diag(s[:d]), v[:, :d].T))

        u_old = u[:, d].copy()
        u_new = u[:, d].copy()
        v_old = v[:, d].copy()
        v_new = v[:, d].copy()

        converged = False
        iter_count = 0

        while not converged:
            iter_count += 1

            u_new = np.dot(X_res, v_new)
            u_new = u_new / np.sqrt(np.sum(u_new * u_new))
            v_new = np.dot(X_res.T, u_new)
            v_new = v_new / np.sqrt(np.sum(v_new * v_new))

            if np.sum(np.sqrt(np.power(v_new - v_old, 2))) < atol:
                converged = True
            elif iter_count == n_iter:
                warn = True
            else:
                u_old = u_new.copy()
                v_old = v_new.copy()

        u[:, d] = u_new
        v[:, d] = v_new
        s[d] = np.dot(u[:, d], np.dot(X_res, v[:, d]))

    if warn:
        print("The algorithm has not converged, but the maximum number of iterations is reached.")

    return u, s, v.T


if __name__ == "__main__":

    # example
    A = np.array([[3, 2, 2], [5, 3, -2]]).T
    n_dim = min(A.shape)
    N, P = A.shape

    u = np.random.normal(size=(N * N)).reshape(N, N)
    v = np.random.normal(size=(P * P)).reshape(P, P)
    s = np.zeros(n_dim)

    u, s, v = svd_2(A, u, v, s, n_dim, 1000000, 1e-12)
    print("Custom SVD implementation:")
    print("Singular values:\n", s)
    print("U:\n", u)
    print("VT:\n", v)
    s_matrix = np.zeros(A.shape)
    for i in range(len(s)):
        s_matrix[i][i] = s[i]
    print("U*Sigma*VT:\n", np.dot(np.dot(u, s_matrix), v))
    print("Original matrix:\n", A)
    print()

    from scipy.linalg import svd

    print("Scipy SVD implementation:")
    U, s, VT = svd(A)
    s_matrix = np.zeros(A.shape)
    for i in range(len(s)):
        s_matrix[i][i] = s[i]
    print("Singular values:\n", s)
    print("U:\n", U)
    print("VT:\n", VT)
    print("U*Sigma*VT:\n", np.dot(np.dot(U, s_matrix), VT))
    print("Original matrix:\n", A)
    print()

    tol = 1e-12
    """
    assert np.allclose(s, s, atol=tol)
    assert np.allclose(u, U, atol=tol)
    assert np.allclose(v, VT, atol=tol)
    """
    assert np.allclose(np.dot(np.dot(u, s_matrix), v), A, atol=tol)

    for i in range(0,10):
        for j in range(0,10):
            A = np.random.rand(i, j)
            n_dim = min(A.shape)
            N, P = A.shape

            u = np.random.normal(size=(N * N)).reshape(N, N)
            v = np.random.normal(size=(P * P)).reshape(P, P)

            s = np.zeros(n_dim)



            u, s, v = svd_2(A, u, v, s, n_dim, 1000000, 1e-12)
            matrix_s=np.zeros(A.shape)
            for k in range(len(s)):
                matrix_s[k][k]=s[k]
            assert np.allclose(np.dot(np.dot(u, matrix_s), v), A, atol=1e-12)
            print(f"Test {i}x{j} passed.")
    print("All tests passed.")
