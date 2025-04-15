import math
import numpy as np
import torch
import random
from numpy.ma.core import array

import number_theory


def correct_speedUp(A, B, C, c, k, primes):

    n = A.shape[0]

    #primes = primes_correction(c, math.sqrt(k), n)
    C_first = correct(A, B, C, c, math.sqrt(k), primes)
    B_transposed = B.T
    A_transposed = A.T
    C_transposed = C_first.T
    C_final = correct(B_transposed, A_transposed, C_transposed, c, math.sqrt(k), primes)
    return C_final.T

def primes_correction(c, k, n):
    T = (c * k * math.log(n, 2)) / (math.log(math.log(n, 2)))
    print("T" + str(T))
    return  number_theory.generate_first_t_primes(T)



def correct(A, B, C, c, k, primes):
    n = np.shape(A)[0]
    C_here = C.clone()

    for p in primes:

        v_vectors = construct_matrices(n, p)

        for i in range(p):

            B_v = torch.matmul(B, v_vectors[i])

            left_side = torch.matmul(A, B_v)
            right_side = torch.matmul(C_here, v_vectors[i].to_dense())


            for j in range(len(left_side)):
                if left_side[j] != right_side[j]:
                    C_here[j] = torch.matmul(A[j:j + 1], B)

    return C_here



def construct_matrices(n, p):
    v_vectors = [torch.zeros(n, dtype=torch.int64) for _ in range(p)]

    for j in range(n):
        index = j % p
        v_vectors[index][j] = 1

    return v_vectors


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

























