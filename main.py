import torch
import os

from flint import fmpz_poly

import approximation_testing
import approximation_testing_two
import correction_algorithmica
import mm_verification
import test_cases
from approximation import compute_summary
import pyperf

from approximation_testing_two import run_group_approximation_test
from benchmarks import correction_c_test, benchmarks_verification, benchmarks_verification_band_matrices, \
    benchmark_matrix_multiplication, benchmarks_verification_change_max_value
from generate_matrices import generate_pair_solution_matrices, generate_and_save_matrices, load_and_save_as_torch
from generate_plots import plot_calculated_results
from mm_verification import verification
from os_correct_zero_test import test_os_correct_zero, test_os_correct_zero_dummy

if __name__ == "__main__":

    test_os_correct_zero_dummy()


