import time

import torch
from numpy.ma.core import nonzero
from numpy.matrixlib.defmatrix import matrix

import number_theory
from all_zeroes import construct_evaluate, all_zeroes, compute_polynomial, check_g, compute_omega, calculate_single_row
from mm_verification import change_to_verification_form_torch, matrix_to_list
from os_matrix_multiplication import updated_values_and_list

# List which contains all the sub-matrices which have nonzero elements
# The last is always the smallest, so reversed logic to the pseudo code
L = []

# modular value of the algo, a prime of size larger than n squared
p: int

# primitive root for this prime
omega: int

# number of non-zeros
t: int

# omega values for q
omegas_q: [int]

# omega values for r
omegas_r: [int]

# number of rows
n: int

# number of columns
l: int


class MatrixProduct:
    def __init__(self, parent,  matrix_A, matrix_B, granularity, row_start, col_start):
        """
        Initialize a submatrix of the product AB with its test values.

        :param matrix_A: The first matrix (2D list or NumPy array).
        :param matrix_B: The second matrix (2D list or NumPy array).
        :param max_granularity: The maximum granularity (upper bound for tau).
        """
        self.parent = parent
        self.row_start = row_start
        self.col_start = col_start
        self.matrix_A = matrix_A
        self.matrix_B = matrix_B
        self.granularity = granularity
        self.actual_granularity = 0
        self.test_values = []  # Preallocated list for test values
        self.nonzero_count = 0  # Number of nonzero entries found in this submatrix
        self.children = None  # Child submatrices (initialized to None until needed)


    def __eq__(self, other):
        if not isinstance(other, MatrixProduct):
            return False

        return (
                self.row_start == other.row_start and
                self.col_start == other.col_start and
                self.rows() == other.rows() and
                self.column() == other.column()
        )


    def single_row_column_product(self):
        if self.rows() == 1 or self.column() == 1:
            return True
        return False


    def set_test_values(self, test_values):
        self.test_values = test_values




    def find_local_i_j(self, global_i, global_j):
        """
        Convert global (i, j) indices into local coordinates within this MatrixProduct.

        :param global_i: row index in the full matrix
        :param global_j: column index in the full matrix
        :return: (local_i, local_j)
        """
        local_i = global_i - self.row_start
        if global_i - self.row_start < 0:
            local_i = global_i

        local_j = global_j
        return local_i, local_j

    def rows(self):
        """
        Return the size of the submatrix as (num_rows, num_columns).
        """
        return len(self.matrix_A[0])

    def column(self):
        return len(self.matrix_A)

    def update_index(self, i, j, value):
        """
        updates the value of the last found nonzero for this matrix product. Gets called by update_list
        :param i:
        :param j:
        :param value:
        :return:
        """
        global n

        self.matrix_A[j + n][i] = value

        return self

    def update_granularity(self):
        """
        Function for the method Update_values_and_list
        Refreshed the granularity by setting it to t or the value of the parent
        :return:
        """
        global n
        global t
        if self.rows() == n:
            self.granularity = t
        else:
            self.granularity = self.parent.granularity
        compute_test_values(self)
        self.actual_granularity = self.granularity

    def get_child_containing(self, i, j):
        """
        Returns the child MatrixProduct that contains the index (i, j).
        If the index is out of bounds or there are no children, returns None.
        """
        i_local = i - self.row_start
        j_local = j - self.col_start
        if self.children is None:
            return None  # No children, so cannot refine selection further.

        mid_row = self.rows() // 2
        mid_col = self.rows() // 2

        if i_local < mid_row:  # Upper half
            if j_local < mid_col:  # Left half
                return self.children[0]
            else:  # Right half
                return self.children[1]
        else:  # Lower half
            if j_local < mid_col:  # Left half
                return self.children[2]
            else:  # Right half
                return self.children[3]

    def split(self):
        """
        Split the sub-matrix into four smaller sub-matrices.
        - `matrix_A` is a list of columns → split each column into two halves (top & bottom).
        - `matrix_B` is a list of rows → split each row into two halves (left & right).
        """
        if self.children is None:
            global t
            new_granularity = min(self.rows() // 2, t)

            mid_row = self.rows() // 2
            mid_col = self.column() // 2

            # Splitting each column in matrix_A into top and bottom halves
            A_top = [lst[:mid_row] for lst in self.matrix_A]  # First half of each list
            A_bottom = [lst[mid_row:] for lst in self.matrix_A]

            # Splitting each row in matrix_B into left and right halves
            B_left = [lst[:mid_row] for lst in self.matrix_B]  # First half of each list
            B_right = [lst[mid_row:] for lst in self.matrix_B]

            self.children = [
                MatrixProduct(self, A_top, B_left, new_granularity, self.row_start, self.col_start),
                MatrixProduct(self, A_top, B_right, new_granularity, self.row_start, self.col_start + mid_row),
                MatrixProduct(self, A_bottom, B_left, new_granularity, self.row_start + mid_row, self.col_start),
                MatrixProduct(self, A_bottom, B_right, new_granularity, self.row_start + mid_row,
                              self.col_start + mid_row),
            ]
            return self.children


def os_matrix_multiplication_mod_p(A, B, number_of_non_zeroes, prime):
    """
    Computing the matrix product AB, if AB contains at most t non-zeros under modular p
    :param A: first matrix
    :param B: second matrix
    :param number_of_non_zeroes: amount of max calculatable non-zeros
    :param prime: prime which defines the field for which the algorithm runs

    :return: calculated matrix product of AB, so C
    """

    # Taking time for benchmarking
    t0 = time.perf_counter()
    timings = {}

    # Determine amount of rows of A
    global n
    n = A.shape[0]
    a_l = A.shape[1]

    # Set the global variables for the prime and calculate and set the primitive root and number of non-zeroes
    global t
    t = number_of_non_zeroes

    global p
    p = prime

    global omega

    primitive_root = number_theory.find_primitive_root(p)
    omega = primitive_root

    # initialize C as zero matrix
    C = torch.zeros((n, n), dtype=torch.int32)

    # create canonical matrix form
    A_n, B_n = change_to_verification_form_torch(A, B, C)


    # Determine amount of columns of A_n
    global l
    l = A_n.shape[1]

    # TODO check if different matrix format is possible
    A_list = matrix_to_list(A_n, l, True)
    B_list = matrix_to_list(B_n, l, False)

    # caluclate the omega values
    global omegas_q
    global omegas_r
    first, last = compute_omega(p, omega, t, l)
    omegas_q = first
    omegas_r = last


    # Verify that verification form has the correct dimensions

    #assert n * 2 == l

    # create matrixProduct of whole matrix
    matrix_product_AB = MatrixProduct(None, A_list, B_list, t, 0, 0)

    t1 = time.perf_counter()

    # calculate test values for global matrix
    compute_test_values(matrix_product_AB)

    # Verify if there are non-zeroes in the code
    if verify_test_values(matrix_product_AB):
        L.append(matrix_product_AB)

    timings["first_check"] = time.perf_counter() - t1
    timings["find_nonzero"] = 0
    timings["calculate_nonzero"] = 0
    timings["update_values_and_list"] = 0

    # while there is no interval with a nonzero
    while L:

        # take the smallest matrix product currently
        smallest_matrix_product = L[-1]

        t2 = time.perf_counter()

        # find the indices of one non-zero pair in the smallest matrix product
        i_global, j_global = find_nonzero(smallest_matrix_product, smallest_matrix_product.row_start, smallest_matrix_product.col_start)

        timings["find_nonzero"] += time.perf_counter() - t2

        t3 = time.perf_counter()

        # calculate the non-zero element for the index i, j for the matrix C and by this for A_n
        # We move it by n because C got concordinate onto A

        A_n, value = calculate_nonzero(A, B, A_n, i_global, j_global, i_global, j_global + a_l)

        timings["calculate_nonzero"] += time.perf_counter() - t3

        t4 = time.perf_counter()

        update_values_and_list(matrix_product_AB, value.item(), i_global, j_global)

        timings["update_values_and_list"] += time.perf_counter() - t4

        # Debugging step not part of the algorithm
        for e in L:
            verify_test_values(e)

    total_time = time.perf_counter() - t0
    return A_n, total_time, timings


def calculate_nonzero(A, B, C, i_a, j_b, i_c, j_c):
    """

    :param A:
    :param B:
    :param C:
    :param i_a:
    :param j_b:
    :param i_c:
    :param j_c:
    :return:
    """
    row = A[i_a, :]
    column = B[:, j_b]
    value = torch.matmul(row, column)
    C[i_c, j_c] = value
    return C, value


def find_nonzero(matrix_product: MatrixProduct, parent_i, parent_j):
    """
    Recursive function for finding a nonzero element in a matrix product
    :param matrix_product: matrix on which gets recursed
    :return: index of the nonzero element
    """
    global L

    # checks if the matrix is already only of size 1
    if matrix_product.single_row_column_product():
        return matrix_product.row_start, matrix_product.col_start

    # splits the matrix
    matrix_product.split()
    children = matrix_product.children

    for child in children:
        compute_test_values(child)

        if verify_test_values(child):
            L.append(child)
            return find_nonzero(child, parent_i, parent_j)

    # If no child has nonzero test values, double granularity and retry
    matrix_product.granularity *= 2
    return find_nonzero(matrix_product, parent_i, parent_j)



def update_values_and_list(
        matrix_product: MatrixProduct,
        value,
        i,
        j
):
    """
    Updating test values and L-membership of all canonical submatrices
    :param matrix_product:
    :param i: row index of the update value
    :param j: column index of the update value
    """
    global L

    # updates the granularity
    matrix_product.update_granularity()

    # update the matrix product at index i, j
    i_local, j_local = matrix_product.find_local_i_j(i, j)
    matrix_product = matrix_product.update_index(i_local, j_local, value)

    # Recalculating needed q and r values for update
    q_old, q_new, r = recalculating_q_and_r_values(matrix_product, i_local, j_local)

    # test values get updated
    old_test_values = matrix_product.test_values
    updated_test_values = update_test_values(old_test_values, q_old, q_new, r)
    matrix_product.set_test_values(updated_test_values)

    # TODO check if correct useasge of not
    # TODO how to address elments in L and how to remove them
    if not verify_test_values(matrix_product):
        delete_from_L(matrix_product)
        return

    if not matrix_product.single_row_column_product():
        child = matrix_product.get_child_containing(i, j)
        #i_local, j_local = child.find_local_i_j(i, j)
        update_values_and_list(child, value, i, j)



#TODO check if test values already got calculated so that we dont recalculate them
def compute_test_values(matrix_product):
    """
    Calculates the test values for the missing omegas
    :param matrix_product:
    """

    global omega
    global p
    global omegas_q
    global omegas_r

    # checks if the test values aren't calculated
    if matrix_product.actual_granularity != matrix_product.granularity:


        # uses the all zero method construct_evaluate to compute through FME the values required
        q_value, r_value = construct_evaluate(matrix_product.matrix_A,
                                              matrix_product.matrix_B,
                                              matrix_product.column(),
                                              omegas_q[matrix_product.actual_granularity:matrix_product.granularity],
                                              omegas_r[matrix_product.actual_granularity:matrix_product.granularity],
                                              matrix_product.actual_granularity,
                                              matrix_product.granularity - matrix_product.actual_granularity,
                                              p)


        # get the already calculated test values and combines them with the already existing
        test_values_calculated = matrix_product.test_values

        test_values = compute_polynomial(q_value, r_value)
        matrix_product.set_test_values(test_values_calculated + test_values)

        # set the actual granularity at the same level as the granularity because the test values got calculated
        matrix_product.actual_granularity = matrix_product.granularity


def verify_test_values(matrix_product):
    """
    Returns true if there is a nonzero element
    :param matrix_product:
    :return:
    """
    return not check_g(matrix_product.test_values)

# TODO CHECK IF WE CAN WORK WITH TENSORS OR DO WE NEED LISTS FOR FLINT
def recalculating_q_and_r_values(matrix_product, i, j):
    """
    Compute the values of the polynomials q_old, q_new, and r at the test points omega^0, omega^1, ..., omega^(tau-1).
    Here:
    - q_old represents the old version of q (before the update).
    - q_new represents the updated version of q (after the change in C[i,j]).
    - r is the polynomial for the columns.
    These values are computed efficiently using fast polynomial evaluation (Proposition 5.5).
    :param matrix_product: corresponding matrix product
    :return: q_old, q_new, r
    """
    global n
    global p
    global omegas_q

    i_local, j_local = matrix_product.find_local_i_j(i, j)

    current_granularity = matrix_product.actual_granularity

    assert current_granularity == len(matrix_product.test_values)
    column = matrix_product.matrix_A[n + j].copy()
    row = matrix_product.matrix_B[n + j].copy()
    q_new = calculate_single_row(column, omegas_q[:current_granularity], current_granularity, p)
    # TODO feels very hacky
    actual_i = i
    column[actual_i] = 0
    q_old = calculate_single_row(column, omegas_q[:current_granularity], current_granularity, p)
    r = calculate_single_row(row, omegas_r[:current_granularity], current_granularity, p)

    return q_old, q_new, r


# TODO check if this update is correct very unsure
def update_test_values(test_values, q_old, q_new, r):
    test_values = test_values.copy()
    assert len(test_values) == len(q_old)
    assert len(test_values) == len(q_new)
    assert len(test_values) == len(r)


    for i, tau in enumerate(test_values):
        test_values[i] += (q_new[i] - q_old[i]) * r[i]


    return test_values

def delete_from_L(matrix_product):
    global L

    # Find the index of the matrix_product in L
    try:
        idx = next(i for i, m in enumerate(L) if m == matrix_product)
    except StopIteration:
        return  # not found

    # Keep only elements before idx and those not descendants
    L = L[:idx] + [m for m in L[idx+1:] if not is_descendant_or_equal(matrix_product, m)]

def is_descendant_or_equal(parent, child):
    # Base case: match
    if parent == child:
        return True
    # Traverse up child’s parent chain
    while child.parent is not None:
        if child.parent == parent:
            return True
        child = child.parent
    return False

def delete_from_L_old(matrix_product):
    global L

    for i, e in enumerate(L):
        if matrix_product.__eq__(e):
            L.pop(i)
            return L

    return L