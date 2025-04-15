import torch
from flint import fmpz_mod_poly_ctx
from numpy.ma.core import shape


import number_theory
from all_zeroes import compute_omega, all_zeroes, construct_evaluate, compute_polynomial
from mm_verification import change_to_verification_form_torch

omegas = []
n_global = 0
tau_values = {}
L = []


class Interval:

    def __init__(self, k, n, matrix, identification, parent):
        self.id = identification
        self.area = (k, n)
        self.parent = parent
        self.tau = (n - k) + 2
        self.children = []
        self.create_children(matrix)
        self.test_values = [[],[]]
        self.matrix = matrix



    def get_area_size(self):


        return self.get_area()[1] - self.get_area()[0]

    def set_tau_value(self, tau):
        self.tau = tau

    def slice_matrix_A(self):
        return self.matrix[self.area[0]:self.area[1] +1, :]



    def slice_matrix_B(self):
        return self.matrix[:, self.area[0]:self.area[1] + 1]

    def get_sliced_matrix(self):
        return self.slice_matrix()

    def create_children(self, matrix):

        if self.get_area_size() == 0:
            return



        midpoint = (self.area[0] + self.area[1]) // 2

        k = self.area[0]
        n = self.area[1]

        child1 = Interval(k=k, n=midpoint, matrix=matrix, identification=0, parent=self)
        child2 = Interval(k=midpoint + 1, n=n, matrix=matrix, identification=1, parent=self)

        self.children = ([child1, child2])

    def add_child(self, child_interval):
        child_interval.parent = self
        self.children.append(child_interval)

    def get_area(self):
        return self.area

    def get_parent(self):
        return self.parent

    def get_tau(self):
        return self.tau

    def get_child_one(self):
        return self.children[0]

    def get_child_two(self):
        return self.children[1]

    def set_test_values(self, test_values, i):
        test_values[i] = test_values

    def get_test_values(self, i):
        return self.test_values[i]

    def get_identification(self):
        return self.id






def compute_matrix_product(A, B, n, t):
    global n_global
    global tau_values
    global omegas
    global L


    n_global = n
    C = torch.zeros((n, n), dtype=torch.int32)
    A_star, B_star = change_to_verification_form_torch(A, B, C)
    n_global = 2 * n_global
    p = number_theory.find_primes((n_global**2)+1, ((n_global+1)**2)+1, 1)[0]
    w = number_theory.find_omega(p)
    omegas = compute_omega(p, w, n_global, n_global)
    f_p = fmpz_mod_poly_ctx(p)

    #initial_omegas = omegas[:t] + omegas[n_global:n_global + t]
    #q_values, r_values = construct_evaluate(A_star, B_star, f_p, n_global, initial_omegas, t, p)
    #test_values = compute_polynomial(q_values, r_values)

    while not all_zeroes(A_star, B_star, p, w, t):
        A_root_interval = Interval(0, n-1, A_star, 0, None)
        B_root_interval = Interval(0, n-1 , B_star,0, None)

        I, J = A_root_interval, B_root_interval
        i, j = find_nonzero(I, J, f_p)
        A_star[i.get_area()[0], n+j.get_area()[0]] = torch.matmul(A[i.get_area()[0], :], B[:, j.get_area()[0]])

        #initial_omegas = omegas[:t] + omegas[n_global:n_global + t]
        #q_values, r_values = construct_evaluate(A_star, B_star, f_p, n_global, initial_omegas, t, p)

        #test_values = compute_polynomial(q_values, r_values)


    return A_star, B_star





def updated_values_and_list(I, J, i, j, t, C, old_vector, f_p):
    if (I.get_tau() == n_global):
        I.set_tau_value(t)
        J.set_tau_value(t)
    else:
        I.set_tau_value(I.get_parent().get_tau_value())
        J.set_tau_value(J.get_parent().get_tau_value())
    compute_update_test_values(C, j, f_p)


def compute_update_test_values(C, j, tau, old_vector, f_p):
    q_part = C[:, j]
    r_part = [0] * n_global
    r_part[j] = -1
    needed_omgas = omegas[:tau] + omegas[n_global:n_global + tau]
    q_values, r_values = construct_evaluate(old_vector, r_part, f_p, n_global, needed_omgas, tau, f_p.modulus())
    q_values_new, r_values_new = construct_evaluate(q_part, r_part, f_p, n_global, needed_omgas, tau, f_p.modulus())
    



def find_nonzero(I, J, f_p):
    if I.get_area_size() == 0 and J.get_area_size() == 0:
        return I, J

    I_one, I_two = I.get_child_one(), I.get_child_two()
    J_one, J_two = J.get_child_one(), J.get_child_two()

    I_parts = [I_one, I_two]
    J_parts = [J_one, J_two]

    print(I_two.get_area())
    for i, I_part in enumerate(I_parts):
        for j, J_part in enumerate(J_parts):
            test_values = compute_test_values(I_part, J_part, f_p)
            print(test_values)
            if not check_g(test_values):
                print(f"Recursing into smaller intervals: I_part={I_part.get_area()}, J_part={J_part}")
                return find_nonzero(I_part, J_part, f_p)
    print(f"Increasing tau: I_tau={I.get_tau()}, J_tau={J.get_tau()}")
    I.set_tau_value(I.get_tau() * 2)
    J.set_tau_value(J.get_tau() * 2)
    print("dsafd")
    return find_nonzero(I, J, f_p)


def select_smallest_submatrix_L(L):
    return L[len(L) - 1]

def compute_test_values(I, J, f_p):
    id = J.get_identification()

    a_sliced = I.slice_matrix_A()
    b_sliced = J.slice_matrix_B()
    initial_omegas = omegas[:I.get_tau()] + omegas[n_global:n_global + J.get_tau()]
    print(shape(a_sliced)[1])
    q_values, r_values = construct_evaluate(a_sliced, b_sliced, f_p, shape(a_sliced)[1], omegas, I.get_tau(), f_p.modulus())
    return compute_polynomial(q_values, r_values)





