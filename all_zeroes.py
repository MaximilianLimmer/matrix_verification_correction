from flint import *
import torch
import time

def compute_omega(p, w, t, l):
    f_p = fmpz_mod_ctx(p)
    result_q = [f_p(w) ** i for i in range(t)]
    result_r = [f_p(w) ** (i * l) for i in range(t)]
    return result_q, result_r

def compute_omega_old(p, w, t, l):
    f_p = fmpz_mod_ctx(p)
    result_q = [f_p(0)] * t
    result_r = [f_p(0)] * t
    w_base = f_p(w)

    w_l = f_p(w ** l)

    result_q[0] = f_p.one()
    result_r[0] = f_p.one()


    for i in range(1, t):
        result_q[i] = w_base
        w_base *= f_p(w)

    for k in range(t):
        result_r[k] = w_l
        w_l *= f_p(w)

    return result_q, result_r

def calculate_single_row(row, omegas, t, p):
    q_k = fmpz_mod_poly(row, fmpz_mod_poly_ctx(p))
    output = q_k.multipoint_evaluate(omegas[:t])
    return output


# TODO study effect of introduction of start value
def construct_evaluate(A, B, l, omega_first, omega_second, start, t, p):

    q_values = [[] for _ in range(t)]
    r_values = [[] for _ in range(t)]

    for k in range(l):

        q_k = fmpz_mod_poly(A[k], fmpz_mod_poly_ctx(p))
        r_k = fmpz_mod_poly(B[k], fmpz_mod_poly_ctx(p))

        # Get the first t elements from a list
        q = zero_check(q_k, omega_first, t)
        r = zero_check(r_k, omega_second, t)

        # TODO kein plan ob das klappt

        for i in range(t):

            q_values[i].append(q[i])
            r_values[i].append(r[i])

    return q_values, r_values


def zero_check(p, omegas, t):
    if p != 0:
        return p.multipoint_evaluate(omegas[:t])
    return [0] * t

def compute_polynomial(q_values, r_values):

    output = [0] * len(q_values)

    for r in range(len(q_values)):
        g = 0
        help_q = q_values[r]
        help_r = r_values[r]

        for i in range(len(help_q)):
            g += help_q[i] * help_r[i]

        output[r] = g

    return output

# f
def check_g(output):
    for x in output:
        if x != 0:
            return False
    return True

def all_zeroes(A, B, p, w, t, n, l):
    timings = {}
    start = time.perf_counter()

    t1 = time.perf_counter()
    omega_first, omega_second = compute_omega(p, w, t, l)
    timings["omega"] = time.perf_counter() - t1
    t2 = time.perf_counter()
    q_values, r_values = construct_evaluate(A, B, l, omega_first, omega_second, 0, t, p)
    timings["construct_eval"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    output = compute_polynomial(q_values, r_values)
    timings["compute_poly"] = time.perf_counter() - t3

    t4 = time.perf_counter()
    result = check_g(output)
    timings["check_g"] = time.perf_counter() - t4

    timings["total"] = time.perf_counter() - start
    return result, timings


def construct_evaluate_zero_check(A, B, f_p, l, omegas, t, p):

    q_values = [[] for _ in range(t)]
    r_values = [[] for _ in range(t)]

    for k in range(l):

        q_k, r_k = None, None

        if not torch.all(A[:, k] == 0):
            q_k = fmpz_mod_poly(A[:, k].tolist(), fmpz_mod_poly_ctx(p))
        if not torch.all(B[k, :] == 0):
            r_k = fmpz_mod_poly(B[k, :].tolist(), fmpz_mod_poly_ctx(p))

        q = q_k.multipoint_evaluate(omegas[:t]) if q_k else [0] * t
        r = r_k.multipoint_evaluate(omegas[t:]) if r_k else [0] * t

        for i in range(t):
            q_values[i].append(q[i])
            r_values[i].append(r[i])

    return q_values, r_values