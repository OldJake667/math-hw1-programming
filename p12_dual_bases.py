import numpy as np

rng = np.random.default_rng(0)

from pathlib import Path


def dual_basis(P, tol=1e-10):
    """Return the rows that extract coordinates in the column basis of P."""
    P = np.asarray(P)
    if P.ndim != 2 or P.shape[0] != P.shape[1] or tol <= 0:
        raise ValueError("expected a square matrix and a positive tolerance")

    n = P.shape[0]
    R = np.hstack([P.astype(float), np.eye(n)])
    # Row reduce [P | I] to get the dual rows, which satisfy R_dual @ P = I.
    for c in range(n):
        pivot = c
        for i in range(c + 1, n):
            if abs(R[i, c]) > abs(R[pivot, c]):
                pivot = i
        if abs(R[pivot, c]) < tol:
            raise np.linalg.LinAlgError("matrix is singular")

        R[[c, pivot]] = R[[pivot, c]]
        R[c] /= R[c, c]
        for i in range(n):
            if i != c:
                R[i] -= R[i, c] * R[c]
                R[i, c] = 0
    return R[:, n:]


def vandermonde(nodes):
    """The k-th row evaluates the monomials at node k."""
    n = len(nodes)
    V = np.ones((n, n))
    for j in range(1, n):
        V[:, j] = V[:, j - 1] * nodes
    return V


def interpolate(nodes, values, x):
    """Evaluate p(x) = sum_k values[k] * L_k(x) at the points x."""
    p = np.zeros_like(x, dtype=float)
    for k, t_k in enumerate(nodes):
        L_k = np.ones_like(x, dtype=float)
        for j, t_j in enumerate(nodes):
            if j != k:
                L_k *= (x - t_j) / (t_k - t_j)
        p += values[k] * L_k
    return p


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P = rng.normal(size=(5, 5))
    rows = dual_basis(P)
    x = rng.normal(size=5)
    coordinates = rows @ x
    assert np.allclose(rows @ P, np.eye(5))
    assert np.allclose(P @ coordinates, x)

    nodes = np.array([-1.0, 0.0, 1.0])
    V = vandermonde(nodes)
    # V @ coeffs(L_j) = e_j, so column j of V^(-1) gives coeffs(L_j).
    coefficients = dual_basis(V)
    expected = np.array([[0, 1, 0], [-0.5, 0, 0.5], [0.5, -1, 0.5]])
    assert np.allclose(V @ coefficients, np.eye(3))
    assert np.allclose(coefficients, expected)
    assert np.allclose(coefficients @ [1, 2, 5], [2, 2, 1])
    print("Coefficients [1, t, t^2] for L_-1, L_0, L_1")
    print(coefficients.T)
    print("Example 8.9 gives t(t - 1)/2, 1 - t^2, t(t + 1)/2.")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout="constrained")
    cases = [
        (lambda t: np.sin(3 * t), -1, 1, 9, r"$\sin(3t)$"),
        (lambda t: 1 / (1 + t**2), -5, 5, 21, r"$1/(1+t^2)$"),
    ]
    for ax, (f, a, b, count, label) in zip(axes, cases):
        nodes = np.linspace(a, b, count)
        values = f(nodes)
        grid = np.linspace(a, b, 1501)
        p = interpolate(nodes, values, grid)
        assert np.allclose(interpolate(nodes, values, nodes), values)

        ax.plot(grid, f(grid), color="#214d72", linewidth=2, label="Function")
        ax.plot(grid, p, color="#ad451d", linestyle="--", label="Interpolant")
        ax.scatter(nodes, values, s=18, color="#214d72", zorder=3, label="Nodes")
        ax.set(title=f"{label}, {count} equally spaced nodes", xlabel="t", ylabel="Value")
        ax.grid(alpha=0.22)
        ax.legend(loc="best", frameon=False)
        print(f"{count} nodes, max grid error = {abs(p - f(grid)).max():.6g}")

    output = Path(__file__).with_name("p12_interpolation.png")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print("Runge's phenomenon: p matches the nodes but oscillates near both ends of [-5, 5].")
