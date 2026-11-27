import numpy as np

rng = np.random.default_rng(0)


def rank_one(u, v, w):
    return np.einsum("i,j,k->ijk", u, v, w)


if __name__ == "__main__":
    T = np.arange(24).reshape(2, 3, 4)
    flat = T.reshape(-1)
    for i in range(2):
        for j in range(3):
            for k in range(4):
                assert T[i, j, k] == flat[12 * i + 4 * j + k]

    mode_0 = T.reshape(2, 12)
    mode_1 = T.transpose(1, 0, 2).reshape(3, 8)
    print("First unfolding")
    print(mode_0)
    print("\nSecond unfolding")
    print(mode_1)
    # The first two rows are independent in both unfoldings.
    for M in (mode_0, mode_1):
        minor = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        assert abs(minor) > 1e-10
        assert np.linalg.matrix_rank(M) == 2
    # The third row in the second unfolding adds no rank.
    assert np.allclose(mode_1[2], 2 * mode_1[1] - mode_1[0])
    print("Both unfoldings have rank 2.")

    # Table 1
    A = rng.normal(size=(3, 4))
    B = rng.normal(size=(4, 2))
    S = rng.normal(size=(4, 4))
    x, y = rng.normal(size=(2, 4))

    assert np.allclose(np.einsum("ij,jk->ik", A, B), A @ B)
    assert np.allclose(np.einsum("ii->", S), np.trace(S))
    assert np.allclose(np.einsum("i,i->", x, y), x @ y)
    assert np.allclose(np.einsum("i,j->ij", x, y), np.outer(x, y))
    assert np.allclose(np.einsum("ij->ji", A), A.T)
    assert np.allclose(np.einsum("i,ij,j->", x, S, y), x @ S @ y)

    A = rng.normal(size=(100, 4, 4))
    B = rng.normal(size=(100, 4, 4))
    traces = np.einsum("bii->b", A)
    products = np.einsum("bij,bjk->bik", A, B)
    for b in range(100):
        assert np.allclose(traces[b], np.trace(A[b]))
        assert np.allclose(products[b], A[b] @ B[b])

    A = rng.normal(size=(3, 4))
    B = rng.normal(size=(2, 5))
    X = rng.normal(size=(4, 5))
    assert np.allclose((A @ X @ B.T).reshape(-1), np.kron(A, B) @ X.reshape(-1))
    print("\nFor 100 x 100 inputs, kron(A, B) stores 100,000,000 entries.")
    print("The Kronecker matvec takes 100^4 multiplications. A @ X @ B.T takes 2 * 100^3.")

    u = rng.normal(size=3)
    v = rng.normal(size=4)
    w = rng.normal(size=5)
    T = rank_one(u, v, w)
    uvT = np.outer(u, v)
    for k, w_k in enumerate(w):
        assert np.allclose(T[:, :, k], w_k * uvT)
