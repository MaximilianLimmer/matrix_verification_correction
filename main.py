import torch
import os

from flint import fmpz_poly

import approximation_testing
import approximation_testing_two
import bench_new
import correction_algorithmica
import mm_verification
import test_cases
from approximation import compute_summary
import pyperf
import math

from approximation_testing_two import run_group_approximation_test
from benchmarks import correction_c_test, benchmarks_verification, benchmarks_verification_band_matrices, \
    benchmark_matrix_multiplication, benchmarks_verification_change_max_value
from generate_matrices import generate_pair_solution_matrices, generate_and_save_matrices, load_and_save_as_torch
from generate_plots import plot_calculated_results
from mm_verification import verification
from os_correct_zero_test import test_os_correct_zero, test_os_correct_zero_dummy

if __name__ == "__main__":

    t_fn_const = lambda size: 2
    t_fn_double = lambda size: size * 2
    t_fn_log = lambda size: max(1, int(math.log2(size)))
    t_fn_sqrt = lambda size: int(math.sqrt(size))
    t_fn_n = lambda size: size // 2

    # Example run:
    bench_new.benchmark_all(c=1, t_fn=t_fn_n)


