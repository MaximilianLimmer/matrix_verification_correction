import torch
import os
import math
from flint import fmpz_poly
from numpy.ma.core import max_val
from tensorboard.compat.tensorflow_stub.dtypes import int32

import approximation_testing
import approximation_testing_two
import bench_new
import correction_algorithmica
import generate_matrices
import mm_verification
import os_correct_zero_test
import test_cases
from approximation import compute_summary
import pyperf
import math

from approximation_testing_two import run_group_approximation_test
from benchmarks import correction_c_test, benchmarks_verification, benchmarks_verification_band_matrices, \
    benchmark_matrix_multiplication, benchmarks_verification_change_max_value
from correction_algorithmica import primes_correction
from generate_matrices import generate_pair_solution_matrices, generate_and_save_matrices, load_and_save_as_torch, \
    create_file_structure_with_rank1_support
from generate_plots import plot_calculated_results
from mm_verification import verification
from os_correct_zero_test import test_os_correct_zero, test_os_correct_zero_dummy

def verification_run_bench():
    t_fn_const = lambda size: 2
    t_fn_double = lambda size: size * 2
    t_fn_log = lambda size: max(1, int(math.log2(size)))
    t_fn_sqrt = lambda size: int(math.sqrt(size))
    t_fn_n = lambda size: size // 2

    # Example run:
    bench_new.benchmark_all(c=2, t_fn=t_fn_log)

def correction_run_bench():
    import torch

    A = torch.load("data_test_int/vandermonde/u_0/A_size_32.pt")  # Replace XX with actual size
    print(A.dtype)
    n = 1000
    k = 4
    c = 1.3841

    t_fn_const = lambda size: 2
    t_fn_double = lambda size: size * 2
    t_fn_log = lambda size: max(1, int(math.log2(size)))
    t_fn_sqrt = lambda size: int(math.sqrt(size))
    t_fn_n = lambda size: size // 2

    bench_new.benchmark_correct_speedUp(c, t_fn_log)

def generate_matrices_nonzeroes():
    nonzero_fn_map = {
        "rank1_sqrt(n)_nonzeroes": lambda n: max(2, int(math.sqrt(size)))  # At least 1 nonzero
    }

    sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]  # or any other sizes you want

    # Loop through each size
    for size in sizes:
        create_file_structure_with_rank1_support(
            matrix_types=["rank1_sqrt(n)_nonzeroes"],  # Include 'rank1_sqrt(n)_nonzeroes' here
            sizes=[size],  # Just one size per call
            max_value=size,  # Max value equals the matrix size
            dtype=torch.int,  # Data type for tensors
            sparsity=0.0,  # Sparsity is not used for rank1_sqrt(n)_nonzeroes
            nonzero_fn_map=nonzero_fn_map  # Pass the nonzero function map
        )

        A = torch.load("data_test_int/rank1_sqrt(n)_nonzeroes/u_0/A_size_16.pt")  # Replace XX with actual size
        print(A)


if __name__ == "__main__":


    n = 512
    l = 512
    max_value = n
    dtype = torch.int32
    matrix_type = "random"
    sparsity = 1
    c = 1
    t = 10

    test_os_correct_zero(n, l, max_value, dtype, matrix_type, sparsity, t)


    A, B, C, C_error = generate_matrices.generate_pair_solution_error_matrices(n, 2*n, n, torch.int32, "random", 1, 5, True)

    print(mm_verification.verification(A, B, C_error, c, t, [], []))
