from fme_correct_verify import mm_verification, os_correct_zero
from generate_matrices import generate_pair_matrices, \
    generate_pair_solution_error_matrices
import torch

from fme_correct_verify.mm_verification import calculate_primes


def test_os_correct_zero(n, l, max_value, dtype, matrix_type, sparsity, t):

    A, B, C, C_error = generate_pair_solution_error_matrices(n, l, max_value, dtype, matrix_type, sparsity, 2, True)
    A_n, B_n = mm_verification.change_to_verification_form_torch(A, B, C_error)
    assert not torch.equal(C, C_error)
    new = torch.matmul(A_n, B_n)
    prime = calculate_primes(n + 1 ,1)
    print(prime)
    solution, total_time, timings = os_correct_zero.os_matrix_multiplication_mod_p(A_n, B_n, t, prime[0].item())
    print(total_time)
    print(timings)
    C_sol = solution[:, -n:]
    print(C_sol)
    print(solution)
    print(solution.shape)
    assert torch.equal(new, C_sol)

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




