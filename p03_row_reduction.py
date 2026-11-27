import numpy as np

rng = np.random.default_rng(0)


def rref(A, tol=1e-10):
    """Return the reduced matrix and its zero-based pivot columns."""
    R = np.array(A, dtype=float)
    if R.ndim != 2 or tol <= 0:
        raise ValueError("expected a matrix and a positive tolerance")

    m, n = R.shape
    pivots = []
    r = 0

    for c in range(n):
        if r == m:
            break

        # Partial pivoting.
        pivot_row = r
        for i in range(r + 1, m):
            if abs(R[i, c]) > abs(R[pivot_row, c]):
                pivot_row = i

        if abs(R[pivot_row, c]) < tol:
            R[r:, c] = 0
            continue

        R[[r, pivot_row]] = R[[pivot_row, r]]
        R[r] /= R[r, c]
        for i in range(m):
            if i != r:
                R[i] -= R[i, c] * R[r]
                R[i, c] = 0

        pivots.append(c)
        r += 1

    return R, pivots


def rank(A):
    _, pivots = rref(A)
    return len(pivots)


def null_space_basis(A):
    R, pivots = rref(A)
    n = R.shape[1]
    basis = []

    for free in range(n):
        if free in pivots:
            continue

        # One basis vector per free variable.
        v = np.zeros(n)
        v[free] = 1
        for r, c in enumerate(pivots):
            v[c] = -R[r, free]
        basis.append(v)

    return basis


def col_space_basis(A):
    A = np.asarray(A)
    _, pivots = rref(A)
    return A[:, pivots]


if __name__ == "__main__":
    A = np.array([[1, 2, -1], [3, -1, 4]])
    R, pivots = rref(A)
    assert np.allclose(R, [[1, 0, 1], [0, 1, -1]])
    assert pivots == [0, 1]
    assert np.allclose(col_space_basis(A), [[1, 2], [3, -1]])
    print("RREF of the first matrix")
    print(R)
    print("Pivot columns:", pivots)

    F = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    R, pivots = rref(F)
    assert np.allclose(R, [[1, 0, -1], [0, 1, 2], [0, 0, 0]])
    assert pivots == [0, 1]
    assert np.allclose(null_space_basis(F), [[1, -2, 1]])
    print("\nRREF of F")
    print(R)

    for _ in range(20):
        A = rng.normal(size=(5, 7))
        basis = null_space_basis(A)
        r = rank(A)
        assert all(np.allclose(A @ v, 0) for v in basis)
        assert r + len(basis) == 7
        assert r == np.linalg.matrix_rank(A)
        assert np.linalg.matrix_rank(col_space_basis(A)) == r

    A = rng.normal(size=(6, 4)) @ rng.normal(size=(4, 8))
    basis = null_space_basis(A)
    assert rank(A) == np.linalg.matrix_rank(A) == 4
    assert len(basis) == np.linalg.matrix_rank(np.array(basis)) == 4
    assert all(np.allclose(A @ v, 0) for v in basis)
    assert np.linalg.matrix_rank(col_space_basis(A)) == 4

    print("\n6 x 8 example: rank 4, nullity 4.")
