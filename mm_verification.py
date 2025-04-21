import numpy as np
import torch
import all_zeroes
import number_theory



def change_to_verification_form(A, B, C):
    I = (-1 * np.eye(np.shape(B)[1])).astype(int)
    return np.hstack((A, C)), np.vstack((B, I))


def change_to_verification_form_torch(A, B, C):
    I = -1 * torch.eye(B.shape[1], dtype=torch.int32)
    AC = torch.hstack((A, C))
    BI = torch.vstack((B, I))
    return AC, BI

def calculate_primes(n, d):
    lower_limit = (n ** 2) + 1
    upper_limit = (2 ** d) * lower_limit
    return number_theory.find_primes_torch(lower_limit, upper_limit, d)

def calculate_primitive_roots(primes):
    roots = []
    for i in primes:
        roots.append(number_theory.find_primitive_root(i.item()))
    return roots

def verification(A, B, C, c, t, primes, omegas):
    import time

    A_n, B_n = change_to_verification_form_torch(A, B, C)
    n = A_n.size(0)
    l = A_n.size(1)
    d = c + 1

    timing_meta = {}

    if len(primes) == 0:
        prime_start = time.perf_counter()
        primes = calculate_primes(n, d)
        timing_meta["prime_calc"] = time.perf_counter() - prime_start
    else:
        timing_meta["prime_calc"] = 0.0

    if len(omegas) == 0:
        omega_start = time.perf_counter()
        omegas = calculate_primitive_roots(primes)
        timing_meta["omega_calc"] = time.perf_counter() - omega_start
    else:
        timing_meta["omega_calc"] = 0.0

    A_list = matrix_to_list(A_n, l, True)
    B_list = matrix_to_list(B_n, l, False)

    all_timings = []
    total_start = time.perf_counter()

    for i in range(d):
        p = primes[i].item()
        w = omegas[i]

        iter_start = time.perf_counter()
        result, timing = all_zeroes.all_zeroes(A_list, B_list, p, w, t, n, l)
        iter_time = time.perf_counter() - iter_start

        timing["iteration_total"] = iter_time
        all_timings.append(timing)

        if not result:
            total_time = time.perf_counter() - total_start
            timing_meta["total"] = total_time
            return False, total_time, all_timings, timing_meta

    total_time = time.perf_counter() - total_start
    timing_meta["total"] = total_time
    return True, total_time, all_timings, timing_meta

def verification_numpy(A, B, C):
    C_dash = np.dot(A, B)
    return np.equal

def verification_torch(A, B, C):
    C_dash = torch.matmul(A, B)
    return torch.equal(C, C_dash)

def compare_matrices(A, B):
    for row1, row2 in zip(A, B):
        for elem1, elem2 in zip(row1, row2):
            if elem1 != elem2:
                return False, elem1, elem2
    return True

def matrix_to_list(A, l, column):
    output = []
    for i in range(l):
        if column:
            output.append(A[:,i].tolist())
        else:
            output.append(A[i,:].tolist())
    return output