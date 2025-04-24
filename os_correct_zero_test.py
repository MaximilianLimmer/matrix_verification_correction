from PIL.DdsImagePlugin import item1

import os_correct_zero
from generate_matrices import generate_pair_solution_matrices, generate_pair_matrices
import torch

from mm_verification import calculate_primes


def test_os_correct_zero(n, l, max_value, dtype, matrix_type, sparsity, t):

    A, B, C = generate_pair_solution_matrices(n, l, max_value, dtype, matrix_type, sparsity)
    prime = calculate_primes((n* n) + 1 ,1)
    print(prime)
    solution = os_correct_zero.os_matrix_multiplication_mod_p(A, B, t, prime[0].item())
    C_sol = solution[:, -n:]
    print(C_sol)
    print(solution)
    assert torch.equal(C, C_sol)

def test_os_correct_zero_dummy():

    n = 8
    t = 64
    A, B = generate_pair_matrices(n, n, n , torch.int32, "random", 1)
    #A, B = torch.ones(n, n, dtype=torch.int32), torch.ones(n, n, dtype=torch.int32)

    C_torch = torch.matmul(A, B)

    prime = calculate_primes(n, 1)
    solution = os_correct_zero.os_matrix_multiplication_mod_p(A, B, t, 107)
    C = solution[:, -n:]

    print(C_torch)
    print(C)
    assert torch.equal(C, C_torch)




