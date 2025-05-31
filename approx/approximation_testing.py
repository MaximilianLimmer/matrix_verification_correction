import random

from approx.approximation import compute_summary, reconstruct_approximation_matrix


def relative_error(approx, true):
    return torch.abs(approx - true) / (true + 1e-9)


def compute_summary_wrapper(A, B, b):
    S = compute_summary(A, B, b)
    print(S)
    return S


def test_basic_correctness():
    A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    B = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
    C_true = torch.matmul(A, B)
    summary = compute_summary_wrapper(A, B, b=2)
    print(summary)

    C_approx = torch.zeros_like(C_true)
    for value, (i, j) in summary:
        C_approx[i, j] = value

    print("\nBasic Correctness Test:")
    print("True Matrix:\n", C_true)
    print("Approximated Summary:\n", C_approx)
    print("Error:\n", relative_error(C_approx, C_true))


import torch


def test_error_analysis(n, b, max_attempts=10):


    A, B, C_true, guaranteed_threshold, guarantee_met = distribute_heavy_hitters_spread(n, b, 10, 300, 30)

    C_1_norm_actual = torch.norm(C_true, p=1, dim=0).max()
    guaranteed_error_bound = C_1_norm_actual / b

    summary = compute_summary_wrapper(A, B, b)

    C_approx = torch.zeros_like(C_true)
    for value, (i, j) in summary:
        C_approx[i, j] = value


    errors = abs(torch.sub(C_true, C_approx))

    heavy_hitters_true = top_b_heaviest_elements(C_true, b)
    correct_hits = len(get_identical_elements_by_coordinates(heavy_hitters_true, summary))

    max_error = torch.max(errors).item()
    mean_error = torch.mean(errors).item()

    guarantee_met = max_error <= guaranteed_error_bound


    print(f"\nGuaranteed Error Bound (|C|_1 / b): {guaranteed_error_bound:.3f}")
    print(f"Max Error: {max_error:.3f} (Guarantee Met: {guarantee_met})")
    print(f"Mean Error: {mean_error:.3f}")
    print(f"Heavy hitters detected correctly: {correct_hits}/{b}")

    return {
        "max_error": max_error,
        "mean_error": mean_error,
        "guarantee_met": guarantee_met,
        "correct_heavy_hitters": correct_hits
    }


def test_zipfian_distribution(n, b):

    A = torch.tensor(torch.randint(1, 100, (n, n)), dtype=torch.float32)
    B = torch.tensor(torch.randint(1, 100, (n, n)), dtype=torch.float32)
    C_true = torch.matmul(A, B)
    summary = compute_summary_wrapper(A, B, b)

    # Extract heaviest entries from true matrix
    heavy_hitters_true = torch.argsort(-C_true.flatten())[:b]
    heavy_hitters_approx = [i * n + j for value, (i, j) in summary]

    correct_hits = len(1 for i in heavy_hitters_approx if i in heavy_hitters_true)


def top_b_heaviest_elements(tensor, b):

    values, indices = torch.topk(tensor.flatten(), b)


    coordinates = [divmod(idx.item(), tensor.shape[1]) for idx in indices]

    return [(val.item(), coord) for val, coord in zip(values, coordinates)]

def are_coordinates_identical(list1, list2):
    coords1 = {coord for _, coord in list1}
    coords2 = {coord for _, coord in list2}
    return coords1 == coords2

def get_identical_elements_by_coordinates(list1, list2):

    coord_to_value = {coord: val for val, coord in list2}

    identical_elements = [(val, coord) for val, coord in list1 if coord in coord_to_value]

    return identical_elements


def distribute_heavy_hitters_spread(n, b, max_attempts, base_value, heavy_multiplier, seed=42):

    torch.manual_seed(seed)
    random.seed(seed)
    attempt = 0

    while attempt < max_attempts:
        attempt += 1
        A = torch.rand(n, n) * 0.1
        B = torch.rand(n, n) * 0.1
        columns = list(range(b))
        rows = list(range(b))

        for i in range(b):
            row, col = rows[i], columns[i]
            A[row, col] = base_value * heavy_multiplier
            B[col, row] = base_value * heavy_multiplier


        C_true = torch.matmul(A, B)

        C_1_norm_actual = torch.norm(C_true, p=1, dim=0).max()
        guaranteed_threshold = C_1_norm_actual / b

        heavy_hitters_true = top_b_heaviest_elements(C_true, b)


        all_satisfy = all(val >= guaranteed_threshold for val, _ in heavy_hitters_true)

        if all_satisfy:

            print(f"\nBase Value: {base_value:.3f}, Multiplier: {heavy_multiplier:.3f}")
            print(f"Actual |C|_1: {C_1_norm_actual:.3f}")
            print(f"Guaranteed Threshold (|C|_1 / b): {guaranteed_threshold:.3f}")
            return A, B, C_true, guaranteed_threshold, True

    print("\nMax attempts reached. Returning last attempt.")
    return A, B, C_true, guaranteed_threshold, False


def test_exact_solution_by_approximation(n, b, max_value):

    A = torch.zeros(n, n)
    B = torch.zeros(n, n)

    verifier_colum = n * [False]
    verifier_row = n * [False]
    for i in range(b):

        colum_rand = random.randint(0, n-1)
        while verifier_colum[colum_rand]:
            colum_rand = random.randint(0, n-1)
        verifier_colum[colum_rand] = True


        row_rand = random.randint(0, n-1)
        while verifier_row[row_rand]:
            row_rand = random.randint(0, n-1)
        verifier_row[row_rand] = True



        value_a = random.randint(1, max_value)
        value_b = random.randint(1, max_value)

        A[row_rand, colum_rand] = value_a
        B[colum_rand, row_rand] = value_b

    solution = torch.matmul(A, B)
    approximation_result = compute_summary(A, B, b)
    approximation_matrix = reconstruct_approximation_matrix(approximation_result, n)

    assert torch.equal(solution, approximation_matrix)

    print("\nExact solution for isolated heavy hitter elements")




def run_all_tests():

    test_exact_solution_by_approximation(100, 100, 100)
    test_error_analysis(n=20, b=15)
    test_error_analysis(n=150, b=50)
    test_error_analysis(n=300, b=100)



