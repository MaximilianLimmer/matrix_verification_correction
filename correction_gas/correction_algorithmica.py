import math
import numpy as np
import torch
import random
import time
from numpy.ma.core import array
from tensorboard.manager import start

import number_theory


def correct_speedUp(A, B, C, c, k, primes):
    timings = {}
    iteration_timings = {}

    if len(primes) == 0:
        start_prime = time.perf_counter()
        n = np.shape(A)[0]
        primes = primes_correction(c, math.sqrt(k), n)
        timings["prime_calculation"] = time.perf_counter() - start_prime


    iteration_timings.setdefault("column_arrangement", 0.0)
    iteration_timings.setdefault("vector_calculation", 0.0)
    iteration_timings.setdefault("matrix_vector_calc", 0.0)

    start = time.perf_counter()


    #primes = primes_correction(c, math.sqrt(k), n)
    C_first, iteration_timings = correct(A, B, C, c, math.sqrt(k), primes, iteration_timings)
    timings["first_iteration"] = time.perf_counter() - start

    tranposition_start = time.perf_counter()
    B_transposed = B.T
    A_transposed = A.T
    C_transposed = C_first.T
    timings["matrix_tranposed"] = time.perf_counter() - tranposition_start

    second_iteration = time.perf_counter()
    C_final, iteration_timings = correct(B_transposed, A_transposed, C_transposed, c, math.sqrt(k), primes, iteration_timings)
    timings["second_iteration"] = time.perf_counter() - second_iteration

    total_time = time.perf_counter() - start


    return C_final.T, total_time, timings, iteration_timings

def primes_correction(c, k, n):
    T = (c * k * math.log(n, 2)) / (math.log(math.log(n, 2)))
    return  number_theory.generate_first_t_primes(T)



def correct(A, B, C, c, k, primes, iteration_timings):
    n = A.shape[0]

    for p in primes:
        t0 = time.perf_counter()
        v_vectors = construct_matrices(n, p)
        iteration_timings["column_arrangement"] += time.perf_counter() - t0

        for i in range(p):
            vector_time = time.perf_counter()
            B_v = torch.matmul(B, v_vectors[i])
            left_side = torch.matmul(A, B_v)
            right_side = torch.matmul(C, v_vectors[i])
            iteration_timings["vector_calculation"] += time.perf_counter() - vector_time

            error_rows = (left_side != right_side).nonzero(as_tuple=True)[0]
            if error_rows.numel() == 0:
                continue

            cols = (v_vectors[i] == 1).nonzero(as_tuple=True)[0]

            t2 = time.perf_counter()
            A_err = A[error_rows, :]
            B_sub = B[:, cols]
            corrected = torch.matmul(A_err, B_sub)
            for idx, row in enumerate(error_rows):
                C[row, cols] = corrected[idx]
            iteration_timings["matrix_vector_calc"] += time.perf_counter() - t2

    return C, iteration_timings




def construct_matrices(n, p):
    v_vectors = torch.zeros((p, n), dtype=torch.float64)
    for j in range(n):
        v_vectors[j % p, j] = 1
    return [v_vectors[i] for i in range(p)]




def is_error_isolated(indices, primes):
    """
    Check if each error index is isolated from all others for at least one prime.
    """
    for im in indices:
        isolated = False
        for p in primes:
            mod_im = im % p
            conflict = False
            for iq in indices:
                if iq != im and iq % p == mod_im:
                    conflict = True
                    break
            if not conflict:
                # This prime isolates im
                isolated = True
                break
        if not isolated:
            # No prime isolates im
            return False
    return True

def determine_error_indices(n, k, c):
    """Determine if the error indices are isolated for a given c."""
    primes = primes_correction(c, k, n)
    indices = set(random.sample(range(1, n + 1), k))

    if is_error_isolated(indices, primes):
        print(f"All errors are isolated for c = {c}.")
        return True
    else:
        print(f"Errors are not isolated for c = {c}.")
        return False

























