import pyperf
from fme_correct_verify import mm_verification
import torch
import os

from generate_matrices import generate_pair_solution_matrices

result = []


def setup_verification(n, d):
    """
    Setup for the nonzero verification by calculating primes and primitve roots
    :param n: size of matrix
    :param d: max number of possible non-zeroes
    :return: primes and primitve rroots
    """
    primes = mm_verification.calculate_primes(n, d)
    primitive_roots = mm_verification.calculate_primitive_roots(primes)
    return primes, primitive_roots

def verification_torch(A, B, C):
    """
    Verification of matrix product using torch by recalculating the product and using the equal method of torch
    :param A: left element of matrix product
    :param B:  right element of matrix product
    :param C: object which gets verified
    :return: True if the element which gets verified is indeed correct
    """
    C_recalculated = torch.matmul(A, B)
    return torch.equal(C_recalculated, C)

def benchmark_func(method, *args):
    """
    Wrapper for the benchmarking of a function for not only getting the result but also storing the results
    :param method: method to be benchmarked
    :param args: parameters of the function
    """
    result.append(method(args))



def verification_test_generated_matrices(benchmark_name, bench_verification_name, size, c, amount_of_non_zeroes, method_for_matrices, *args):
    """
    Verification has the Parameters (matrix A,
                                    matrix B,
                                    matrix ot verify C,
                                    c which is the upper bound on the largest element in A and B with size * c,
                                    t as the maximum number of non-zeroes so that C can get correctly verified,
                                    primes the set of primes for which modulo the test is run,
                                    primitve roots for these primes
    Benchmarks this function according to the algorithm given in the paper
    On Nondeterministic Derandomization of Freivalds’ Algorithm:
    Consequences, Avenues and Algorithmic Progress
    mentioned in chapter 5 non_zero test
    :param benchmark_name: name of method which runs for benchmark
    :param bench_verification_name: name of verification benchmark
    :param size: size rows and column in A and B
    :param amount_of_non_zeroes: amount of verifiable non-zeroes in C
    :return: the benchmark runner
    """

    runner = pyperf.Runner()

    matrices_list = method_for_matrices(args)

    primes, primitive_roots = setup_verification(size, amount_of_non_zeroes)

    for i, (A, B, C) in enumerate(matrices_list):

        runner.bench_func(benchmark_name, benchmark_func, mm_verification.verification, A, B, C, c, amount_of_non_zeroes, primes, primitive_roots)
        runner.bench_func(bench_verification_name, benchmark_func, verification_torch, A, B, C)

        global result
        assert(result[0] == result[1])
        result = []

    return runner



def save_matrices(matrix_type, size, sparsity, num_matrices, max_value, dtype=torch.int64):
    """
    Generate and save matrices for a specific type, size, and sparsity.
    """
    for i in range(num_matrices):
        if matrix_type == 'sparse_0.1':
            A, B, C = generate_pair_solution_matrices(size, size, max_value, dtype, matrix_type, sparsity)
        else:
            A, B, C = generate_pair_solution_matrices(size, size, max_value, dtype, matrix_type)

        # Create directory
        folder = f"matrices/{matrix_type}/size_{size}"
        if sparsity is not None:
            folder += f"/sparsity_{sparsity}"
        os.makedirs(folder, exist_ok=True)


        torch.save(A, f"{folder}/A_{i}.pt")
        torch.save(B, f"{folder}/B_{i}.pt")
        torch.save(C, f"{folder}/C_{i}.pt")


def generated_matrix_test_verification(n, c, t):
    matrix_types = [
        "random",
        "sparse_0.1",
        "toeplitz",
        "diagonal",
        "gaussian",
        "identity",
        "symmetric",
        "triangular",
        "ones",
        "laplacian",
        "hilbert",
        "permutation",
        "band",
        "nilpotent"
    ]
    for x in matrix_types:
        A, B, C = generate_pair_solution_matrices(n, n, n, torch.int32, x, 1)
        value = mm_verification.verification(A, B, C, c, t, [], [])
        assert value == True
