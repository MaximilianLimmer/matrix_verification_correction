import torch

def freivalds(A, B, C, k):

    n = A.shape[0]
    for _ in range(k):
        r = torch.randint(0, 2, (n, 1))
        Br = torch.matmul(B, r)
        left = torch.matmul(A, Br)
        right = torch.matmul(C, r)
        if not torch.equal(left, right):
            return False
    return True