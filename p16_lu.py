import numpy as np

rng = np.random.default_rng(0)

from time import perf_counter


def lu(A, tol=1e-10):
    """Return the unit lower and upper factors L and U without row swaps."""
    U = np.array(A, dtype=float)
    if U.ndim != 2 or U.shape[0] != U.shape[1] or tol <= 0:
        raise ValueError("expected a square matrix and a positive tolerance")

    n = U.shape[0]
    L = np.eye(n)
    for k in range(n - 1):
        if abs(U[k, k]) < tol:
            raise np.linalg.LinAlgError(f"pivot {k} is too small for LU without swaps")
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k + 1:] -= L[i, k] * U[k, k + 1:]
            U[i, k] = 0

    return L, U


def solve_lu(L, U, b, tol=1e-10):
    """Solve L U x = b with one or more right-hand sides."""
    L, U, b = np.asarray(L), np.asarray(U), np.asarray(b)
    if U.ndim != 2 or U.shape[0] != U.shape[1] or tol <= 0:
        raise ValueError("expected a square upper factor and a positive tolerance")
    n = U.shape[0]
    if L.shape != U.shape or b.ndim not in (1, 2) or b.shape[0] != n:
        raise ValueError("expected matching factors and right-hand sides")

    y = b.astype(float)
    for i in range(n):
        for j in range(i):
            y[i] -= L[i, j] * y[j]

    x = y.copy()
    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < tol:
            raise np.linalg.LinAlgError(f"pivot {i} is too small for back substitution")
        for j in range(i + 1, n):
            x[i] -= U[i, j] * x[j]
        x[i] /= U[i, i]

    return x


def inverse(A):
    L, U = lu(A)
    return solve_lu(L, U, np.eye(len(U)))


def update_gram_inverse(G_inv, row, tol=1e-10):
    """Apply Sherman-Morrison for G_new = G + row row^T."""
    left = G_inv @ row
    right = row @ G_inv
    denominator = 1 + row @ left
    if abs(denominator) < tol:
        raise np.linalg.LinAlgError("rank-one update is singular")
    return G_inv - (left[:, None] * right[None, :]) / denominator


if __name__ == "__main__":
    A = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]])
    b = np.array([5, 13, 31])
    L, U = lu(A)
    assert np.allclose(L, [[1, 0, 0], [2, 1, 0], [4, 3, 1]])
    assert np.allclose(U, [[2, 1, 1], [0, 1, 1], [0, 0, 2]])
    assert np.allclose(L @ U, A)
    x = solve_lu(L, U, b)
    assert np.allclose(x, [1, 2, 1])
    print("Example 6.107")
    print("L =", L, "\nU =", U, "\nx =", x)

    for _ in range(20):
        A = rng.normal(size=(6, 6))
        b = rng.normal(size=6)
        L, U = lu(A)
        x = solve_lu(L, U, b)
        assert np.allclose(L @ U, A)
        assert np.allclose(A @ x, b)
        assert np.allclose(x, np.linalg.solve(A, b))

    A = np.array([[1e-17, 1], [1, 1]])
    b = np.array([1, 2])
    # Lower tol to let the tiny pivot through.
    L, U = lu(A, tol=1e-18)
    x = solve_lu(L, U, b, tol=1e-18)
    rows = [1, 0]
    L_swapped, U_swapped = lu(A[rows])
    x_swapped = solve_lu(L_swapped, U_swapped, b[rows])
    expected = np.linalg.solve(A, b)
    assert not np.allclose(A @ x, b)
    assert np.allclose(x_swapped, expected)
    print("\nTiny pivot, no swaps:", x)
    print("Swapping the two rows of A and b:", x_swapped)
    print("The tiny pivot makes the multiplier 1e17, so rounding wipes out the smaller terms.")

    X = rng.normal(size=(200, 50))
    new_rows = rng.normal(size=(100, 50))
    G = X.T @ X
    initial_inverse = inverse(G)

    start = perf_counter()
    updated = initial_inverse
    for row in new_rows:
        updated = update_gram_inverse(updated, row)
    update_seconds = perf_counter() - start

    start = perf_counter()
    running_gram = G.copy()
    for row in new_rows:
        running_gram += row[:, None] * row[None, :]
        recomputed = inverse(running_gram)
    every_row_seconds = perf_counter() - start

    X_final = np.vstack([X, new_rows])
    G_final = X_final.T @ X_final
    start = perf_counter()
    final_inverse = inverse(G_final)
    final_only_seconds = perf_counter() - start

    assert np.allclose(updated, recomputed)
    assert np.allclose(updated, final_inverse)
    assert np.allclose(updated, np.linalg.inv(G_final))
    assert np.allclose(G_final @ updated, np.eye(G_final.shape[0]))

    print("\n100 rank-one updates")
    print(f"Sherman-Morrison after every row: {update_seconds:.6f} s")
    print(f"LU inverse from scratch after every row: {every_row_seconds:.6f} s")
    print(f"LU inverse on the final Gram matrix only: {final_only_seconds:.6f} s")
    print(f"Largest difference from the final inverse: {abs(updated - final_inverse).max():.3e}")
    print("One update takes quadratic work in the matrix size. Elimination takes cubic work.")
