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


def verification(A, B, C,  c, t, primes, omegas):

    A_n, B_n = change_to_verification_form_torch(A, B, C)

    n = A_n.size(0)
    print(n)
    l = A_n.size(1)
    d = c + 1

    if len(primes) == 0:
        primes = calculate_primes(n, d)

    if len(omegas) == 0:
        omegas = calculate_primitive_roots(primes)

    A_list = matrix_to_list(A_n, l, True)
    B_list = matrix_to_list(B_n, l, False)

    for i in range(d):
        print(i)
        p = primes[i].item()
        w = omegas[i]


        if not all_zeroes.all_zeroes(A_list, B_list, p, w, t, n, l):
            print("Verification output:")
            print("False")
            return False
    print("Verification output:")
    print("True")
    return True

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