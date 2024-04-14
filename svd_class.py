import numpy as np
import pickle as pkl
import os
class svd_2():
    def __init__(self, A, n_iter, atol,demo=False, name=""):
        self.A = A
        if not demo:
            self.u = np.random.normal(size=(A.shape[0] * A.shape[0])).reshape(A.shape[0], A.shape[0])
            self.v = np.random.normal(size=(A.shape[1] * A.shape[1])).reshape(A.shape[1], A.shape[1])
            self.s = np.zeros(min(A.shape))
        else:
            if os.path.exists(f"./cache/{name}_u.pkl") and os.path.exists(f"./cache/{name}_v.pkl") and os.path.exists(f"./cache/{name}_s.pkl"):
                with open(f"./cache/{name}_u.pkl","rb") as f:
                    self.u = pkl.load(f)
                with open(f"./cache/{name}_v.pkl","rb") as f:
                    self.v = pkl.load(f)
                with open(f"./cache/{name}_s.pkl","rb") as f:
                    self.s = pkl.load(f)

                if self.u.shape[0]!=A.shape[0] or self.v.shape[0]!=A.shape[1]:
                    self.u = np.random.normal(size=(A.shape[0] * A.shape[0])).reshape(A.shape[0], A.shape[0])
                    self.v = np.random.normal(size=(A.shape[1] * A.shape[1])).reshape(A.shape[1], A.shape[1])
                    self.s = np.zeros(min(A.shape))
            else:
                self.u = np.random.normal(size=(A.shape[0] * A.shape[0])).reshape(A.shape[0], A.shape[0])
                self.v = np.random.normal(size=(A.shape[1] * A.shape[1])).reshape(A.shape[1], A.shape[1])
                self.s = np.zeros(min(A.shape))

        self.n_dim = min(A.shape)
        self.n_iter = n_iter
        self.atol = atol
        self.demo = demo
        self.name = name

    def __call__(self, *args, **kwargs):
        X_res = self.A.copy()
        warn = False

        for d in range(self.n_dim):
            if d >= 1:
                X_res = self.A - np.dot(self.u[:, :d], np.dot(np.diag(self.s[:d]), self.v[:, :d].T))

            u_old = self.u[:, d].copy()
            u_new = self.u[:, d].copy()
            v_old = self.v[:, d].copy()
            v_new = self.v[:, d].copy()

            converged = False
            iter_count = 0

            while not converged:
                iter_count += 1

                u_new = np.dot(X_res, v_new)
                u_new = u_new / np.sqrt(np.sum(u_new * u_new))
                v_new = np.dot(X_res.T, u_new)
                v_new = v_new / np.sqrt(np.sum(v_new * v_new))

                if np.sum(np.sqrt(np.power(v_new - v_old, 2))) < self.atol:
                    converged = True
                elif iter_count == self.n_iter:
                    warn = True
                else:
                    u_old = u_new.copy()
                    v_old = v_new.copy()

            self.u[:, d] = u_new
            self.v[:, d] = v_new
            self.s[d] = np.dot(self.u[:, d], np.dot(X_res, self.v[:, d]))

        if warn:
            print("The algorithm has not converged, but the maximum number of iterations is reached.")

        if not os.path.exists("./cache"):
            os.mkdir("./cache")
        with open(f"./cache/{self.name}_u.pkl","wb") as f:
            pkl.dump(self.u,f)
        with open(f"./cache/{self.name}_v.pkl","wb") as f:
            pkl.dump(self.v,f)
        with open(f"./cache/{self.name}_s.pkl","wb") as f:
            pkl.dump(self.s,f)

        return self.u, self.s, self.v.T


if __name__ == "__main__":

    # example
    A = np.array([[3, 2, 2], [5, 3, -2]]).T


    u, s, v = svd_2(A, 1000000, 1e-12)()
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



            u, s, v = svd_2(A, 1000000, 1e-12)()
            matrix_s=np.zeros(A.shape)
            for k in range(len(s)):
                matrix_s[k][k]=s[k]
            assert np.allclose(np.dot(np.dot(u, matrix_s), v), A, atol=1e-12)
            print(f"Test {i}x{j} passed.")

    print("All tests passed.")
