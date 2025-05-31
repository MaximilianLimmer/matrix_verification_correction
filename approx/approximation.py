import time

import torch
import math
import numpy as np
from numpy.ma.extras import column_stack
import heapq

from sympy.benchmarks.bench_meijerint import timings


def reconstruct_approximation_matrix(S, n):
    print(len(S))
    C = torch.zeros((n, n))
    for value, flat_index in S:
        row = flat_index[0] % n
        col = flat_index[1] %  n
        if 0 <= row < n and 0 <= col < n:
            C[row, col] = value
        else:
            print(f"Warning: Index ({row}, {col}) is out of bounds and will be ignored.")
    return C


def compute_summary(A, B, b):



    timings = {}
    timings["sorting"] = 0
    timings["find_b_plus_largest_elements"] = 0
    timings["positional_sort"] = 0
    timings["merge"] = 0

    n = A.size(0)
    t_a = time.perf_counter()
    row_order = ordered_list_numpy(A.T)
    column_order = ordered_list_numpy(B)
    timings["pre_sort"] = time.perf_counter() - t_a
    S = []
    t0 = time.perf_counter()
    for i in range(n):

        u = row_order[i*n:(i+1)*n]
        v = column_order[i*n:(i+1)*n]

        t1 = time.perf_counter()
        u_sorted = sorted(u, reverse=True)
        v_sorted = sorted(v, reverse=True)
        timings["positional_sort"] += time.perf_counter() - t1

        t2 = time.perf_counter()
        L_and_plus_one= KMaxCombinations(u_sorted, v_sorted , b + 1)

        timings["find_b_plus_largest_elements"] += time.perf_counter() -t2

        c = L_and_plus_one.pop()[0]
        L = L_and_plus_one
        L = decrease_by_c(L, c, 1)

        t3 = time.perf_counter()
        L = positional_sort(L, n)
        print(L)
        timings["positional_sort"] += time.perf_counter() - t3

        t4 = time.perf_counter()
        S = merge_lists(S, L)
        if len(S) > b+1:
            S = sorted(S, reverse=True)
            x = S[b+1][0]
            S = decrease_by_c(S[:b], x, 1)
        timings["merge"] += time.perf_counter() - t4

        t5 = time.perf_counter()
        #positional_sort(S, n)
        timings["positional_sort"] += time.perf_counter() - t5

        L = []
    total_time = time.perf_counter() - t0
    print(total_time)
    print(timings)
    return S, total_time, timings


def merge_lists(S, L):

    S_dict = {coord: val for val, coord in S}

    for val, coord in L:
        if coord in S_dict:
            S_dict[coord] += val  # Add values when coordinates match
        else:
            S_dict[coord] = val  # Add new entry if coordinate isn't in S

    # Convert dictionary back to a list and return it
    return [(val, coord) for coord, val in S_dict.items()]

def combine(S, L, b, n):
    if len(S) == 0:
        return L

    merged = []
    i, j = 0, 0

    while i < len(S) and j < len(L):
        if i < len(S) and j < len(L) and equal(S[i], L[j], n):
            merged.append((S[i][0] + L[j][0], S[i][1]))
            i += 1
            j += 1
            continue


        if j < len(L) and (i >= len(S) or compare(S[i], L[j], n)):
            merged.append(L[j])
            j += 1
        elif i < len(S):
            merged.append(S[i])
            i += 1

    while i < len(S):
        merged.append(S[i])
        i += 1

    while j < len(L):
        merged.append(L[j])
        j += 1

    return merged


def KMaxCombinations(a, b, k):
    # Sorting the arrays.

    n = len(a)
    output = []



    # Using a max-heap.
    pq = []
    heapq.heapify(pq)
    heapq.heappush(pq, (-a[0][0]  * b[0][0], (0, 0), (a[0][1], b[0][1])))
   # print((-a[0][0] * b[0][0], (0, 0), (a[0][1], b[0][1])))


    # Using a set.
    my_set = set()
    my_set.add((0, 0))

    for count in range(k):

        #  tuple format (sum, (i, j)).
        temp = heapq.heappop(pq)

        here = -temp[0]
        output.append((here, temp[2]))


        i = temp[1][0]
        j = temp[1][1]

        if i < n - 1:

            sum = a[i + 1][0] * b[j][0]
            old_i = a[i + 1][1]
            old_j = b[j][1]
            old = (old_i, old_j)
            temp1 = (i + 1, j)


            if (temp1 not in my_set):
                heapq.heappush(pq, (-sum, temp1, old))
                my_set.add(temp1)

        if j < n - 1:
            sum = a[i][0] * b[j + 1][0]
            old_i = a[i][1]
            old_j = b[j + 1][1]
            old = (old_i, old_j)

            temp1 = (i, j + 1)


            if (temp1 not in my_set):
                heapq.heappush(pq, (-sum, temp1, old))
                my_set.add(temp1)

    return output


def find_entry_of_rank_b_brute(u, v, b):
    # Ensure u and v are sorted in descending order
    assert all(u[i] >= u[i + 1] for i in range(len(u) - 1))
    assert all(v[i] >= v[i + 1] for i in range(len(v) - 1))

    # Generate all pairs (i, j) and their corresponding products
    products = []
    for i in range(len(u)):
        for j in range(len(v)):
            products.append((u[i][0] * v[j][0], i, j))

    # Sort products in descending order
    products.sort(reverse=True, key=lambda x: x[0])

    # Return the b-th largest element (indexing from 0, so b+1 for 1-based rank)
    return products[b]


def find_b_largest_entries(u, v, c, n):
    i = 0
    j = n - 1  # Start with the last element in v
    L = []

    while i < n and j >= 0:
        if u[i][0] * v[j][0] > c:
            # Add all valid (i, k) pairs for k <= j
            L.extend([(u[i][0] * v[k][0], (u[i][1], v[k][1])) for k in range(j + 1)])
            i += 1  # Move to the next row
        else:
            j -= 1  # Decrease column pointer to find larger products

    return L


def ordered_list_numpy(matrix):
    mat_np = matrix.numpy().flatten()
    indices = np.arange(mat_np.size)
    return list(zip(mat_np.tolist(), indices.tolist()))

def list_log_negate(u):
    output = []
    for x in u:
        if x[0] == 0:
            output.append((0, x[1:]))
        else:
            output.append((math.log(x[0]), x[1:]))

    return output

def select_values(indices, u, v, c, n):
    L = []
    for (x, y) in indices:
        L.append((u[x] * v[y] - c, x*n + y))
    return L

def positional_sort(tuples, n):
    return sorted(tuples, key=lambda x: x[1][0] * n + x[1][1])

def decrease_by_c(L, c, p):
    return list(((x[0] + c), x[p]) for x in L)

def equal(x, y, n):
    if x[1][0] * n + x[1][1] == y[1][0] * n + y[1][1]:
        return True
    return False

def compare(x, y, n):
    if x[1][0] * n + x[1][1] >=  y[1][0] * n + y[1][1]:
        return True
    return False
