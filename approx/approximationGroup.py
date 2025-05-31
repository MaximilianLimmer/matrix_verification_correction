
import math
import numpy as np
import torch
import time
from functools import wraps
from flint import fmpz_poly
from sympy.benchmarks.bench_meijerint import timings

from number_theory import find_primes_for_task



def approximation(A, B, b):
    """
    Calls methods which runs the group approximation algorithm
    Let A, B be real n × n matrices and C = AB their product.
    :param A: first matrix
    :param B: second matrix
    :param b: amount of element which get approximated
    :return: approximation of matrix product
    """
    timings = {}
    t0 = time.perf_counter()

    n = A.shape[0]
    l = A.shape[1]

    t1 = time.perf_counter()
    P = find_primes_for_task(b, n * l)
    print(P)
    timings["find_primes"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    Q, G, L, timing_group = compute_group(A, B, P, n, l)
    timings["compute_group"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    output, timing_check = check_candidates(A, B, G, Q, P, L, n)
    timings["check_candidates"] = time.perf_counter() - t3

    timings["total_time"] = time.perf_counter() - t0
    return output, timings, timing_group, timing_check




def check_candidates(A, B, G, Q, P, L, n):
    """
    Calculates the bit lists of the elements which contributes the most out of the outer product and calculates its in C
    :param A: matrix A
    :param B: matrix B
    :param G: stores total value
    :param Q: stores values for bits
    :param P: list of primes
    :param L: number of bits for the representation of the matrix
    :param n: size of matrix
    :return: Q and G
    """
    timing = {}
    output_c = torch.zeros((n, n))
    X = Q.shape[0]
    Y = Q.shape[1]
    K = torch.zeros((X, Y, L))
    t0 = time.perf_counter()
    for j in range(len(P)):
        p = P[j]
        for m in range(p):


            total_weight = Q[j][m].item()
            bit_list = calculate_bit_list(G[j][m], L, total_weight)

            K[j][m] = bit_list
    indices_list = []
    timing["construct_bit_list"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    timing["construct_indices"] = 0
    timing["calculate_values"] = 0
    for j in range(len(P)):
        p = P[j]
        for m in range(p):
            t1 = time.perf_counter()
            i_index, j_index = find_index(K[j][m], int(L/2))
            indices_list.append((i_index, j_index))
            timing["construct_indices"] += time.perf_counter() - t1

            t2 = time.perf_counter()
            output_c = check_value_torch(A, B, output_c, i_index, j_index, 100)
            timing["calculate_values"] += time.perf_counter() - t2

    #unique_tuples = list(dict.fromkeys(indices_list))
    return output_c, timing



def calculate_bit_list_new(G, L, total_weight):
    bit_list = []
    for k in range(L):
        kth_bit_one_weight = G[k].item()
        kth_bit_zero_weight = total_weight - kth_bit_one_weight
        if kth_bit_one_weight >= kth_bit_zero_weight:
            bit_list.append(1)
        else:
            bit_list.append(0)
    return torch.tensor(bit_list)

def calculate_bit_list(G, L, total_weight):
    """
    Constructs the bit list by comparing if the elements with zero or one at the kth bit are larger
    and by this construct bits list for heaviest element
    :param G: list with weight for bits
    :param L: amount of bits
    :param total_weight: total weights of all elements
    :return: bit list of element which contributes the most
    """
    bit_list = []
    counter = 0
    for k in range(L):

        kth_bit_one_weight = G[k].item()
        kth_bit_zero_weight = total_weight - kth_bit_one_weight

        if np.sign(kth_bit_one_weight) != np.sign(kth_bit_zero_weight):
            # Choose the group matching the sign of total_weight.
            bit_sign = 1 if np.sign(kth_bit_one_weight) == np.sign(total_weight) else 0
            bit_list.append(bit_sign)

        else:
            # If both groups have the same sign, choose based on magnitude.
            bit_sign = 1 if abs(kth_bit_one_weight) >= abs(kth_bit_zero_weight) else 0
            bit_list.append(bit_sign)
    bit = torch.tensor(bit_list)
    return bit



def find_index(t, ell):
    """
    Constructs the index out of the bit representation
    :param t: list of bits
    :param ell: amount of bits
    :return: indices i and j
    """
    i_index = 0
    j_index = 0
    first = t[:ell]
    last = t[ell:]
    i_list = first.tolist()
    j_list = last.tolist()

    for i in range(ell):
        if i_list[i] == 1:
            i_index += 2 ** i
        if j_list[i] == 1:
            j_index += 2 ** i

    return i_index, j_index

def check_value(i, j, A, B):
    """
    Calculates value of matrix C for indices
    :param i: i index
    :param j: j index
    :param A:
    :param B:
    :return: returns value at place (i, j)
    """
    row = A[int(i), :]
    column = B[:, int(j)]

    if 0 == torch.dot(row, column).item():
        return False
    return True



def compute_group(A, B, primes, n, l):
    """
        Calculates the polynomial representation of the rows and columns to calculate the outer product
        and by this the total contribution of the outer product and also for each bit the value if all values
        with bit k set to zero get set to 0
        :param A: matrix A
        :param B: matrix B
        :param n: size of matrix
        :param n: size of matrix
        :return: G and Q, G stores total value, Q stores values for bits

        """

    timing = {}
    timing["FFT_Modular_Convolution"] = 0
    timing["FFT_Modular_Convolution_Update"] = 0
    timing["nullification_of_vectors"] = 0
    timing["calculate_bitmask"] = 0
    timing["apply_bitmask"] = 0
    L, Q, G = construct_data_structures(primes, n)

    t0 = time.perf_counter()
    bitmask = compute_bitmasks(int(L//2), n)
    timing["calculate_bitmask"] = time.perf_counter() - t0

    for i in range(l):
        a = A[:, i]
        b = B[i, :]
        print(i)
        for j in range(len(primes)):

            p = primes[j]


            t1 = time.perf_counter()
            coe_p_ab_dash = compute_polynomial_contribution(a, b, p)
            timing["FFT_Modular_Convolution"] += time.perf_counter() - t1

            for m in range(p):
                Q[j][m] = Q[j][m].item() + int(coe_p_ab_dash[m])

            for k in range(L):
                #a_mod = a.clone()
                #b_mod = b.clone()


                if k < L/2:
                    t2 = time.perf_counter()
                    a_coeffs = apply_bitmask(a, bitmask[k])
                    timing["apply_bitmask"] += time.perf_counter() - t2

                    t3 = time.perf_counter()
                    coe_p_ab_dash_b = compute_polynomial_contribution(a_coeffs, b, p)
                    timing["FFT_Modular_Convolution_Update"] += time.perf_counter() - t3


                else:
                    t4 = time.perf_counter()
                    b_coeffs = apply_bitmask(b, bitmask[k - int(L//2)])
                    timing["apply_bitmask"] += time.perf_counter() - t4

                    t5 = time.perf_counter()
                    coe_p_ab_dash_b = compute_polynomial_contribution(a, b_coeffs, p)
                    timing["FFT_Modular_Convolution_Update"] += time.perf_counter() - t5

                for m in range(p):
                    G[j][m][k] = G[j][m][k].item() + int(coe_p_ab_dash_b[m])

    return Q, G, L, timing


def construct_data_structures(primes, n):
    ell = math.ceil(math.log(n, 2))
    X = len(primes)
    Y = primes[-1]
    L = 2 * ell
    Q = torch.zeros((X, Y))
    G = torch.zeros((X, Y, L))
    return L, Q, G



def compute_polynomial_contribution(a, b, p):
    p_a = construct_pa(a, p)
    p_b = construct_pb(b, p)
    p_ab = p_a * p_b
    p_ab_dash = div_mod_operation_b(p_ab, p)
    coe_p_ab_dash = p_ab_dash.coeffs()
    return fill_up(coe_p_ab_dash, p)

def fill_up(coeffs, p):
    if p > len(coeffs):
        return coeffs + [0] * (p - len(coeffs))
    return coeffs



def set_bit():
    return

# TODO
def div_mod_operation(p_ab, p):
    output = [0] * p
    coeffs = p_ab.coeffs()
    for x in range(len(coeffs)):
        index = x % p
        output[index] += coeffs[x]
    return fmpz_poly(output)


def div_mod_operation_b(p_ab, p):
    coeffs = p_ab.coeffs()
    output = [0] * p

    for x in range(len(coeffs)):
        index = x % p  # Degree reduction modulo p
        output[index] += coeffs[x] # Accumulate coefficients
    return fmpz_poly(output)




# TODO
def construct_pa(a, p):
    n = a.shape[0]
    output = [0] * p
    for i in range(n):
        index = (n * i) % p
        output[index] += a[i].item()
    return fmpz_poly(output)


def construct_pb(b, p):
    """
    Construct polynomial of column vector a by
    :param b:
    :param p:
    :return:
    """
    n = b.shape[0]
    output = [0] * p
    for i in range(n):
        index = i % p
        output[index] += b[i].item()
    return fmpz_poly(output)

def check_value_torch(A, B, C,  i, j, threshold):
    """
    Calculates the C values for some indices and can verify if the value is larger then a
    certain threshold
    :param A:
    :param B:
    :param C:
    :param i:
    :param j:
    :param threshold:
    :return:
    """

    row = A[int(i), :]
    column = B[:, int(j)]
    value = torch.matmul(row, column).item()
    if value != 0:
        C[i, j] = torch.dot(row, column).item()
    # Check if the value exceeds the threshold
    return C








def zero_out_elements_final(k, coefficient):
    """
    Checks if the kth bit of an index is 0 and if so sets the value to zero
    """
    #coefficient = coefficient.clone()
    for i, x in enumerate(coefficient):
        if (i & (1 << k)) == 0:  # Check if the k-th bit is 0
            coefficient[i] = 0
    return coefficient

def compute_bitmasks(max_k, list_length):
    """
    Precompute bitmasks for all k values up to max_k.
    """
    bitmasks = {}
    for k in range(max_k + 1):
        bitmask = torch.tensor([(i & (1 << k)) != 0 for i in range(list_length)], dtype=torch.bool)
        bitmasks[k] = bitmask
    return bitmasks

def apply_bitmask(coefficient, bitmask):
    """
    Applies a precomputed bitmask to zero out elements where the k-th bit is 0.
    """
    coefficient = coefficient.clone()
    coefficient[~bitmask] = 0  # Zero out elements where the bitmask is False
    return coefficient































































