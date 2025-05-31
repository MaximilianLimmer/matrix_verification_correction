from random import randint
import sys
import torch
from sympy.polys.factortools import fmpz_poly
import cProfile, pstats

import approximationGroup
from approximationGroup import construct_pa, construct_pb, div_mod_operation_b, zero_out_elements_final, \
    calculate_bit_list, find_index, compute_group, check_candidates
from generate_matrices import create_matrix_with_errors_torch, generate_pair_solution_matrices
from fme_correct_verify.mm_verification import change_to_verification_form_torch
from number_theory import find_primes_for_task


def diagonal_test(n, b, e):
    A = torch.zeros((n, n))
    B = torch.zeros((n, n))
    for i in range(e):
        k = randint(1, 100)
        A[i, i] = k
        B[i, i] = k
    C = torch.matmul(A, B)
    output = approximationGroup.approximation(A, B, b)

    assert torch.equal(C, output)

def test_approximation_group(n, b, e):
    A, B, C = generate_pair_solution_matrices(n, n, n, torch.int64, "random", 1)
    C = torch.matmul(A, B)
    C_error = create_matrix_with_errors_torch(C, e, 100, True)
    A_n, B_n = change_to_verification_form_torch(A, B, C_error)

    output = approximationGroup.approximation(A_n, B_n, b)

    difference = (C - C_error)

    assert (torch.equal(output, difference))

    return output

def test_approximate_exact(n):
    A = torch.randint(1, n, size=(n, n))
    B = torch.randint(1, n, size=(n, n))
    C = torch.matmul(A, B)

    output = approximationGroup.approximation(A, B, n ** n)
    assert torch.equal(C, output)
    return True

def test_construct_pa():
    a = torch.tensor([1, 2, 3, 4])
    p = 3
    expected_output = [1 + 4, 2, 3]  # Because (n * i) % p for n=4: 0%3=0, 4%3=1, 8%3=2, 12%3=0
    result = approximationGroup.construct_pa(a, p).coeffs()
    assert result == expected_output, f"Expected {expected_output}, got {result}"

def test_div_mod_operation():
    p_ab = fmpz_poly([1, 2, 3, 4, 5])  # Represents 1 + 2x + 3x^2 + 4x^3 + 5x^4
    p = 3
    expected_output = [1 + 4, 2 + 5, 3]  # Coefficients grouped modulo 3
    result = approximationGroup.div_mod_operation(p_ab, p).coeffs()
    assert result == expected_output, f"Expected {expected_output}, got {result}"

def test_polynomial_multiplication():
    p_a = fmpz_poly([1, 2])  # 1 + 2x
    p_b = fmpz_poly([3, 4])  # 3 + 4x
    expected_product = fmpz_poly([3, 10, 8])  # 3 + 10x + 8x^2
    result = p_a * p_b
    assert result == expected_product, f"Expected {expected_product}, got {result}"



def test_div_mod_operation_b():
    # Test Case 1
    p_ab = fmpz_poly([1, 2, 3, 4, 5])  # 1 + 2x + 3x^2 + 4x^3 + 5x^4
    p = 3
    expected_output = fmpz_poly([5, 7, 3])  # 5 + 7x + 3x^2
    result = approximationGroup.div_mod_operation_b(p_ab, p)
    assert result == expected_output, f"Test Case 1 failed: expected {expected_output}, got {result}"

    # Test Case 2
    p_ab = fmpz_poly([0, 0, 0, 7, 8])  # 0 + 0x + 0x^2 + 7x^3 + 8x^4
    p = 2
    expected_output = fmpz_poly([8, 7])  # 0 + 7x + 8x^2
    result = approximationGroup.div_mod_operation_b(p_ab, p)
    assert result == expected_output, f"Test Case 2 failed: expected {expected_output}, got {result}"



def test_polynomial_contribution():
    # Input vectors
    a = torch.tensor([1, 2, 3, 4], dtype=torch.int)  # Column vector
    b = torch.tensor([5, 6, 7, 8], dtype=torch.int)  # Row vector
    p = 3  # Prime number

    # Expected output for compute_polynomial_contribution
    # pa = [5, 2, 3], pb = [13, 6, 7]
    # pa * pb = [5*13, 5*6 + 2*13, 5*7 + 2*6 + 3*13, 2*7 + 3*6, 3*7]
    #         = [65, 30 + 26, 35 + 12 + 39, 14 + 18, 21]
    #         = [65, 56, 86, 32, 21]

    expected_coefficients = [97, 77, 86]

    # Test compute_polynomial_contribution
    coefficients = approximationGroup.compute_polynomial_contribution(a, b, p)
    assert coefficients == expected_coefficients, "compute_polynomial_contribution failed"

def test_zeroing_of_elements():
    coefficient = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int)
    k = 0  # Zero out elements where the 1st bit is 0
    output = approximationGroup.zero_out_elements_final(k, coefficient)
    assert output.tolist() == [0, 2, 0, 4, 0, 6, 0, 8], "zero_out_elements_final failed"

def test_polynomial_construction_large():
    a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    b = torch.tensor([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    p = 11
    p_a = construct_pa(a, p)
    p_b = construct_pa(b, p)
    assert len(p_a.coeffs()) <= p, "p_a degree should be less than p"
    assert len(p_b.coeffs()) <= p, "p_b degree should be less than p"

# Test 2: Polynomial Multiplication with Non-Uniform Data
def test_polynomial_multiplication_non_uniform():
    a = torch.tensor([1, 0, -1, 2, 0])
    b = torch.tensor([0, 3, 0, -2, 1])
    p = 7
    p_a = construct_pa(a, p)
    p_b = construct_pb(b, p)
    p_ab = p_a * p_b
    assert len(p_ab.coeffs()) <= 2 * p - 1, "p_ab degree should be less than 2p-1"

# Test 3: Degree Reduction with High-Degree Polynomials
def test_degree_reduction_high_degree():
    a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    b = torch.tensor([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    p = 11
    p_a = construct_pa(a, p)
    p_b = construct_pb(b, p)
    p_ab = p_a * p_b
    p_ab_dash = div_mod_operation_b(p_ab, p)
    assert len(p_ab_dash.coeffs()) <= p, "p_ab_dash degree should be less than p"

# Test 4: Bitwise Operations with Complex Bit Patterns
def test_bitwise_operations_complex():
    coefficient = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    k = 2
    modified_coefficient = zero_out_elements_final(k, coefficient)
    assert modified_coefficient[0] == 0, "Element with k-th bit 0 should be zeroed out"
    assert modified_coefficient[3] == 0, "Element with k-th bit 0 should be zeroed out"

# Test 5: Bit List Construction with Skewed Weights
def test_bit_list_construction_skewed():
    G = torch.tensor([100, 1, 1, 1, 1])
    L = 5
    total_weight = 104
    bit_list = calculate_bit_list(G, L, total_weight)
    assert bit_list[0] == 1, "First bit should be 1 (dominant weight)"
    assert bit_list[1] == 0, "Second bit should be 0 (non-dominant weight)"

# Test 6: Index Calculation with Large Matrices
def test_index_calculation_large():
    t = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    ell = 5
    i_index, j_index = find_index(t, ell)
    assert i_index == 21, "i_index should be 21"
    assert j_index == 10, "j_index should be 10"

# Test 7: Full Algorithm with Known Output
def test_full_algorithm_known_output():
    A = torch.tensor([[1, 2], [3, 4]])
    B = torch.tensor([[5, 6], [7, 8]])
    b = 2
    primes = find_primes_for_task(b, A.shape[0] * A.shape[1])
    Q, G, L = compute_group(A, B, primes, A.shape[0])
    output = check_candidates(A, B, G, Q, primes, L, A.shape[0])
    expected_output = torch.tensor([[19, 22], [43, 50]])
    assert torch.allclose(output, expected_output), "Output does not match expected result"



# Run all tests
def run_tests():
    test_polynomial_construction_large()
    test_polynomial_multiplication_non_uniform()
    test_degree_reduction_high_degree()
    test_bitwise_operations_complex()
    test_bit_list_construction_skewed()
    test_index_calculation_large()
    #test_full_algorithm_known_output()


def run_group_approximation_test():
    run_tests()
    test_construct_pa()
    test_div_mod_operation()
    test_polynomial_multiplication()
    test_div_mod_operation_b()
    test_polynomial_contribution()
    test_zeroing_of_elements()

    test_approximation_group(64, 1, 1)
    #diagonal_test(64, 4, 4)
    #test_approximation_group(128, 2, 1)
    #test_approximation_group(8, 2, 2)
    #test_approxima
    #tion_group(32, 5, 4)
    return True

def benchmark_try():
    profiler = cProfile.Profile()
    profiler.enable()

    test_approximation_group(64, 1, 1)

    profiler.disable()
    stats = pstats.Stats(profiler, stream=sys.stdout).sort_stats('cumtime')
    print(stats.print_stats())