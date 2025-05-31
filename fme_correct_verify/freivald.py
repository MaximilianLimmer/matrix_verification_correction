import torch


def freivalds(A, B, C, k, modulus=None):
    """
    Probabilistic verification of AB = C using Freivalds' algorithm.

    Args:
        A: (n x n) torch.int32
        B: (n x n) torch.int32
        C: (n x n) torch.int32
        k: number of repetitions
        modulus: if not None, use arithmetic mod p

    Returns:
        True if all tests passed, False otherwise
    """
    n = A.size(0)
    ty = A.dtype
    for _ in range(k):
        r = torch.randint(0, 2, (n, 1), dtype=ty)

        Br = B @ r
        ABr = A @ Br
        Cr = C @ r

        if modulus is not None:
            ABr %= modulus
            Cr %= modulus

        if not torch.equal(ABr, Cr):
            return False
    return True


def naive_matrix_multiplication(A, B):
    m, n = A.shape
    n2, p = B.shape
    assert n == n2
    result = torch.zeros((m, p), dtype=A.dtype)

    for i in range(m):
        Ai = A[i]
        for j in range(p):
            sum_ = 0
            for k in range(n):
                sum_ += Ai[k] * B[k, j]
            result[i, j] = sum_
    return result