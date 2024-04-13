import numpy as np
from scipy.optimize import fsolve

# Abandoned attempt at implementing SVD with eigenvalues and eigenvectors
class SVD:
    def __init__(self, A):
        self.A = A
        self.U, self.VT = self.calculate_singular_vectors()
        self.singular_values = self.sv()
        self.eigenvalues = np.sqrt(self.singular_values)
        self.sigma_matrix = self.sigma_matrix()

    def calculate_singular_vectors(self):
        # Calculate U
        eigvals_U, eigvecs_U = np.linalg.eig(np.dot(self.A, self.A.T))
        idx_U = np.argsort(eigvals_U)[::-1]
        U = eigvecs_U[:, idx_U]
        for row in range(U.shape[1]):
            U[:, row] = -U[:, row] if U[0, row] > 0 else U[:, row]

        # Calculate V^T
        eigvals_VT, eigvecs_VT = np.linalg.eig(np.dot(self.A.T, self.A))
        idx_VT = np.argsort(eigvals_VT)[::-1]
        VT = eigvecs_VT[:, idx_VT].T
        for row in range(VT.shape[1]):
            VT[:, row] = -VT[:, row] if VT[0, row] > 0 else VT[:, row]

        return U, VT

    def sv(self):
        singular_values = []
        shape = self.A.shape
        step = 0.5 # guess step size for fsolve
        guess = 0
        while len(singular_values) < min(shape):
            diag_one = np.eye(self.A.shape[0]) if A.shape[0] <= A.shape[1] else np.eye(self.A.shape[1])

            def f(x):
                return np.linalg.det(self.A @ self.A.T - x * diag_one) if A.shape[0] <= A.shape[1] else np.linalg.det(
                    self.A.T @ self.A - x * diag_one)


            sol = fsolve(f, guess)
            if not any(np.isclose(sol[0], sv) for sv in singular_values):
                singular_values.append(sol[0])
            guess += step
        singular_values = [np.sqrt(sv) for sv in singular_values]
        return np.sort(np.array(singular_values))[::-1]
    def sigma_matrix(self):
        sv = self.sv()
        sorted_singular_values = np.sort(sv)[::-1]
        sigma_matrix = np.zeros(self.A.shape)
        for i in range(len(sorted_singular_values)):
            sigma_matrix[i][i] = sorted_singular_values[i]
        return sigma_matrix





if __name__ == "__main__":
    # example
    A = np.array([[3, 2, 2], [2, 3, -2], [2, -2, 3]])
    c_svd = SVD(A)
    singular_values = c_svd.singular_values
    sigma_matrix = c_svd.sigma_matrix

    print("Custom c_svd implementation:")
    print("Singular values:\n", singular_values)
    print("Sigma matrix:\n", sigma_matrix)
    print("U:\n", c_svd.U)
    print("VT:\n", c_svd.VT)
    print("U*Sigma*VT:\n", np.dot(np.dot(c_svd.U, sigma_matrix), c_svd.VT))
    print("Original matrix:\n", A)


    from scipy.linalg import svd


    print("\nScipy SVD implementation:")
    U, s, VT = svd(A)
    a=np.zeros(A.shape)
    for i in range(len(s)):
        a[i][i]=s[i]
    print("Singular values:\n", s)
    print("Sigma matrix:\n", a)
    print("U:\n", U)
    print("VT:\n", VT)
    print("U*Sigma*VT:\n", np.dot(np.dot(U, a), VT))
    print("Original matrix:\n", A)


    print("U comparison:", c_svd.U)
    print("VT comparison:", U)
    print("Sigma comparison:", c_svd.VT)
    print(VT)


    atol = 1e-12
    assert np.allclose(c_svd.singular_values, s, atol=atol)
    assert np.allclose(c_svd.U, U, atol=atol)
    assert np.allclose(c_svd.VT, VT, atol=atol)
    assert np.allclose(np.dot(np.dot(c_svd.U, c_svd.sigma_matrix), c_svd.VT), A, atol=atol)


