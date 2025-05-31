from fme_correct_verify import mm_verification
from fme_correct_verify.mm_verification import calculate_primes, change_to_verification_form_torch
from fme_correct_verify.os_correct_zero import os_matrix_multiplication_mod_p
from fme_correct_verify.os_correct_zero import os_matrix_multiplication_mod_p
from correction_gas.correction_algorithmica import correct_speedUp
from correction_gas.correction_algorithmica import correct_speedUp
from approx.approximation import compute_summary, reconstruct_approximation_matrix  # adjust import

from fme_correct_verify.freivald import freivalds, naive_matrix_multiplication  #
import re
from fme_correct_verify.os_correct_zero import os_matrix_multiplication_mod_p  # Adjust the import as needed
import os
import torch
import csv
from approx.approximationGroup import approximation  # your approximation function
import generate_matrices
import math
import time
import os, csv, torch
from fme_correct_verify.mm_verification import verification  # replace as needed



def benchmark_group_testing_mm(output_file="bench_group_testing_verified_b_log(n).csv"):


    BASE_DIR = "../data_test_int"
    already = "band"
    # Updated TYPES list without the 'rank1' type
    TYPES = [ "band", "diagonal", "identity", "nilpotent", "ones",
              "permutation", "random", "random_max_value_n", "random_signed",
              "sparse_0.1", "sparse_signed", "symmetric", "toeplitz", "triangular",
              "vandermond_poly_n^{2}"

    ]

    US = ["u_0", "u_1"]
    sizes = [128]



    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    # Write CSV header once
    if not os.path.exists(output_file):
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "type", "u", "size", "b", "total_time",
                "find_primes", "compute_group", "check_candidates",
                "fft_time", "fft_update", "apply_bitmask",
                "construct_bits", "index_calc", "value_calc"
            ])

    for matrix_type in TYPES:
        print(f"Testing type: {matrix_type}")
        partial_results = []

        for u in US:
            dir_path = os.path.join(BASE_DIR, matrix_type, u)
            if not os.path.isdir(dir_path):
                continue

            for size in sizes:
                try:
                    print(f"  Size: {size}")
                    b_val = int(math.sqrt(size))
                    print(f"    b = {b_val}")

                    A = torch.load(os.path.join(dir_path, f"A_size_{size}.pt")).to(dtype=torch.int64)
                    B = torch.load(os.path.join(dir_path, f"B_size_{size}.pt")).to(dtype=torch.int64)
                    C = torch.load(os.path.join(dir_path, f"C_size_{size}.pt")).to(dtype=torch.int64)

                    # Inject error
                    C_error = generate_matrices.create_matrix_with_errors_torch(C, b_val, size, True)
                    t_sol = C - C_error

                    # Verification form
                    A_n, B_n = mm_verification.change_to_verification_form_torch(A, B, C_error)

                    #print("Nonzero entries in t_sol:", torch.count_nonzero(t_sol))

                    # Run the approximation
                    approx_C, timing_total, timing_group, timing_check = approximation(A_n, B_n, b_val)

                    # Extract recovered C from approximation
                    C_sol = approx_C[:, -size:]

                    assert torch.equal(C_sol, t_sol), f"Verification failed at size {size}"

                    partial_results.append([
                        matrix_type, u, size, b_val,
                        f"{timing_total['total_time']:.6f}",
                        f"{timing_total.get('find_primes', 0):.6f}",
                        f"{timing_total.get('compute_group', 0):.6f}",
                        f"{timing_total.get('check_candidates', 0):.6f}",
                        f"{timing_group.get('FFT_Modular_Convolution', 0):.6f}",
                        f"{timing_group.get('FFT_Modular_Convolution_Update', 0):.6f}",
                        f"{timing_group.get('apply_bitmask', 0):.6f}",
                        f"{timing_check.get('construct_bit_list', 0):.6f}",
                        f"{timing_check.get('construct_indices', 0):.6f}",
                        f"{timing_check.get('calculate_values', 0):.6f}"
                    ])

                    print(f"    Done. Time: {timing_total['total_time']:.4f}s")

                except Exception as e:
                    print(f"    Error on {matrix_type}, {u}, size {size}: {e}")

        # Save partial results after each matrix type
        with open(output_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(partial_results)

        print(f"✓ Partial results saved after type: {matrix_type}")

    print(f"✓ Benchmarking complete. Final results in {output_file}")


def benchmark_all(c, t_fn, primes=[], omegas=[], output_file="delete.csv"):


    BASE_DIR = "../data_test_int"
    TYPES = ["band", "diagonal", "identity", "nilpotent", "ones",
             "permutation", "random", "random_max_value_n", "random_signed",
             "sparse_0.1", "sparse_signed", "symmetric", "toeplitz", "triangular",
             "vandermond_poly_n^{2}"

             ]
    US = ["u_0", "u_1"]

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        # header
        writer.writerow([
            "type", "u", "size", "t", "result", "total_time",
            "prime_calc", "omega_calc",
            "omega", "construct_poly", "FME", "compute_poly",
            "check_g", "internal_total", "iteration_total"
        ])

        for matrix_type in TYPES:
            print(f"Processing {matrix_type}")
            rows = []
            dir_base = os.path.join(BASE_DIR, matrix_type)
            for u in US:
                path = os.path.join(dir_base, u)
                if not os.path.isdir(path):
                    continue
                files = os.listdir(path)
                sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


                for size in sizes:
                    A = torch.load(os.path.join(path, f"A_size_{size}.pt")).to(dtype=torch.int64)
                    B = torch.load(os.path.join(path, f"B_size_{size}.pt")).to(dtype=torch.int64)
                    C = torch.load(os.path.join(path, f"C_size_{size}.pt")).to(dtype=torch.int64)
                    B = B.t()
                    is_upper = torch.all(A == torch.triu(A))
                    sparsity = (A == 0).sum().item() / A.numel()
                    print(is_upper)
                    print(sparsity)
                    t_val = t_fn(size)
                    result, total_time, timing_data, timing_meta = verification(
                        A, B, C, c=c, t=t_val, primes=primes, omegas=omegas
                    )
                    print(result)
                    m = timing_meta
                    t = timing_data[0] if timing_data else {}

                    rows.append([
                        matrix_type, u, size, t_val, result, f"{total_time:.6f}",
                        f"{m.get('prime_calc', 0):.6f}", f"{m.get('omega_calc', 0):.6f}",
                        f"{t.get('omega', 0):.6f}", f"{t.get('construct_poly', 0):.6f}",
                        f"{t.get('FME', 0):.6f}", f"{t.get('compute_poly', 0):.6f}",
                        f"{t.get('check_g', 0):.6f}", f"{t.get('total', 0):.6f}",
                        f"{t.get('iteration_total', 0):.6f}"
                    ])
                    print(f"  size={size} → {total_time:.3f}s")

            # write & flush after each type
            writer.writerows(rows)
            f.flush()

    print(f"Benchmarking complete. Results saved to {output_file}")




def benchmark_torch(output_file="benchmark_torch_matmul_int.csv", use_allclose=False, atol=1e-6):
    BASE_DIR = "../data_test_int"
    TYPES = ["band", "diagonal", "identity", "nilpotent", "ones",
             "permutation", "random", "random_max_value_n", "random_signed",
             "sparse_0.1", "sparse_signed", "symmetric", "toeplitz", "triangular",
             "vandermond_poly_n^{2}"

             ]
    US = ["u_0", "u_1"]
    results = []

    for matrix_type in TYPES:
        for u in US:
            dir_path = os.path.join(BASE_DIR, matrix_type, u)
            if not os.path.isdir(dir_path):
                continue

            files = os.listdir(dir_path)
            sizes = sorted({int(f.split("_")[-1].split(".")[0]) for f in files if f.startswith("A_size_")})

            for size in sizes:
                A = torch.load(os.path.join(dir_path, f"A_size_{size}.pt")).to(dtype=torch.int64)
                B = torch.load(os.path.join(dir_path, f"B_size_{size}.pt")).to(dtype=torch.int64)
                #C = torch.load(os.path.join(dir_path, f"C_size_{size}.pt")).to(dtype=torch.float64)
                print(size)
                start = time.perf_counter()
                result = torch.matmul(A, B)
                elapsed = time.perf_counter() - start
                print(elapsed)

                results.append({
                    "type": matrix_type,
                    "u": u,
                    "size": size,

                    "time": elapsed
                })

    # Save to CSV
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "u", "size", "time"])
        for r in results:
            writer.writerow([r["type"], r["u"], r["size"], f"{r['time']:.6f}"])

    print(f"torch.equal benchmark complete. Saved to {output_file}")


def benchmark_os(c, t_fn, primes=[], omegas=[], output_file="bench_os_random_4096.csv"):


    BASE_DIR = "../data_test_int"
    TYPES = ["band", "diagonal", "identity", "nilpotent", "ones",
             "permutation", "random", "random_max_value_n", "random_signed",
             "sparse_0.1", "sparse_signed", "symmetric", "toeplitz", "triangular",
             "vandermond_poly_n^{2}"

             ]  # Only one type now
    US = ["u_0", "u_1"]
    RESULTS = []

    for matrix_type in TYPES:
        print(matrix_type)
        for u in US:
            dir_path = os.path.join(BASE_DIR, matrix_type, u)
            if not os.path.isdir(dir_path):
                continue

            files = os.listdir(dir_path)
            sizes = sorted({int(re.findall(r'\d+', f)[0]) for f in files if f.startswith("A_size_")})
            sizes = [128]
            for size in sizes:
                print(size)
                A = torch.load(os.path.join(dir_path, f"A_size_{size}.pt"))
                B = torch.load(os.path.join(dir_path, f"B_size_{size}.pt"))
                C = torch.load(os.path.join(dir_path, f"C_size_{size}.pt"))
                C= torch.zeros((size, size), dtype=torch.int32)
                # Set number_of_non_zeroes and prime here
                number_of_non_zeroes =  int(math.ceil((math.sqrt(size)))) # Example for this calculation
                prime = calculate_primes((size) + 1 ,1)

                print("here")

                C_help = torch.matmul(A, B)
                t_val = t_fn(size)

                # Call the new function
                A_n, total_time, timings = os_matrix_multiplication_mod_p(A, B, number_of_non_zeroes, prime[0].item())

                C_sol = A_n[:, -size:]

                assert torch.equal(C_help, C_sol)

                RESULTS.append({
                    "type": matrix_type,
                    "u": u,
                    "size": size,
                    "t": t_val,
                    "result": "success",  # Adjust depending on whether the multiplication is successful
                    "total_time": total_time,
                    "timings": timings
                })
                print(total_time)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "type", "u", "size", "t", "result", "total_time",
            "first_check", "find_nonzero", "calculate_nonzero", "update_values_and_list"
        ])
        for r in RESULTS:
            t = r["timings"]
            writer.writerow([
                r["type"], r["u"], r["size"], r["t"], r["result"], f"{r['total_time']:.6f}",
                f"{t.get('first_check', 0):.6f}", f"{t.get('find_nonzero', 0):.6f}",
                f"{t.get('calculate_nonzero', 0):.6f}", f"{t.get('update_values_and_list', 0):.6f}"
            ])

    print(f"Benchmarking complete. Results saved to {output_file}")

def benchmark_all_os_mm(c, t_fn, primes=[], output_file="bench_os_mm_all_zero_t=n_triangular.csv"):


    BASE_DIR = "../data_test_int"
    TYPES = ["band", "diagonal", "identity", "nilpotent", "ones",
             "permutation", "random", "random_max_value_n", "random_signed",
             "sparse_0.1", "sparse_signed", "symmetric", "toeplitz", "triangular",
             "vandermond_poly_n^{2}"

             ]
    US = ["u_0", "u_1"]

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "type", "u", "size", "t", "total_time",
            "first_check", "find_nonzero", "calculate_nonzero", "update_values_and_list",
            "match"
        ])

        for matrix_type in TYPES:
            print(matrix_type)
            for u in US:
                dir_path = os.path.join(BASE_DIR, matrix_type, u)
                if not os.path.isdir(dir_path):
                    continue

                sizes = [4096]
                for size in sizes:
                    try:
                        print(f"{matrix_type} | {u} | size: {size}")
                        t_val = t_fn(size)

                        A = torch.load(os.path.join(dir_path, f"A_size_{size}.pt"))
                        B = torch.load(os.path.join(dir_path, f"B_size_{size}.pt"))
                        C = torch.load(os.path.join(dir_path, f"C_size_{size}.pt"))
                        C_error = generate_matrices.create_matrix_with_errors_torch(C, t_val, size, True)

                        A_n, B_n = mm_verification.change_to_verification_form_torch(A, B, C_error)
                        t_sol = C - C_error

                        primes = calculate_primes(size + 1, 1)
                        if not primes:
                            continue
                        prime = primes[0].item()

                        result_matrix, total_time, timing_data = os_matrix_multiplication_mod_p(A_n, B_n, t_val, prime)
                        C_sol = result_matrix[:, -size:]
                        print(total_time)
                        print(timing_data)
                        match = torch.equal(C_sol, t_sol)

                        writer.writerow([
                            matrix_type, u, size, t_val, f"{total_time:.6f}",
                            f"{timing_data.get('first_check', 0):.6f}",
                            f"{timing_data.get('find_nonzero', 0):.6f}",
                            f"{timing_data.get('calculate_nonzero', 0):.6f}",
                            f"{timing_data.get('update_values_and_list', 0):.6f}",
                            match
                        ])
                        f.flush()
                    except Exception as e:
                        print(f"Error on {matrix_type} {u} size {size}: {e}")
                        continue

    print(f"Benchmarking complete. Results saved to {output_file}")


def benchmark_correct_speedUp(c, k_fn, output_file="bench_correct_sqrt_big.csv"):


    BASE_DIR = "../data_test_int"
    TYPES = ["band", "diagonal", "identity", "nilpotent", "ones",
             "permutation", "random", "random_max_value_n", "random_signed",
             "sparse_0.1", "sparse_signed", "symmetric", "toeplitz", "triangular",
             "vandermond_poly_n^{2}"

             ]
    US = ["u_0", "u_1"]
    RESULTS = []

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "type", "u", "size", "k", "total_time",
            "prime_calculation", "first_iteration", "matrix_tranposed", "second_iteration",
            "column_arrangement", "vector_calculation", "matrix_vector_calc"
        ])

        for matrix_type in TYPES:
            print(matrix_type)
            print(matrix_type)
            for u in US:
                dir_path = os.path.join(BASE_DIR, matrix_type, u)
                if not os.path.isdir(dir_path):
                    continue

                files = os.listdir(dir_path)
                sizes =  [4096*2]

                for size in sizes:
                    print(size)

                    A = torch.load(os.path.join(dir_path, f"A_size_{size}.pt"))
                    B = torch.load(os.path.join(dir_path, f"B_size_{size}.pt"))
                    C = torch.load(os.path.join(dir_path, f"C_size_{size}.pt"))
                    A = A.to(dtype=torch.float64)
                    B = B.to(dtype=torch.float64)
                    C = C.to(dtype=torch.float64)
                    test_copies = C.clone()
                    number_errors = k_fn(size)
                    max_value = size
                    C_error = generate_matrices.create_matrix_with_errors_torch(C, number_errors, max_value, True)


                    k_val = k_fn(size)

                    result, total_time, timings, iteration_timings = correct_speedUp(A, B, C_error, c=c, k=k_val, primes=[])
                    # Verifies correctness of solution
                    #
                    assert torch.equal(test_copies, result)
                    RESULTS.append({
                        "type": matrix_type,
                        "u": u,
                        "size": size,
                        "k": k_val,
                        "total_time": total_time,
                        "timings": timings,
                        "iteration_timings": iteration_timings
                    })
                    print(total_time)

                    # Write results after every iteration
                    t = timings
                    i = iteration_timings
                    writer.writerow([
                        matrix_type, u, size, k_val, f"{total_time:.6f}",
                        f"{t.get('prime_calculation', 0):.6f}", f"{t.get('first_iteration', 0):.6f}",
                        f"{t.get('matrix_tranposed', 0):.6f}", f"{t.get('second_iteration', 0):.6f}",
                        f"{i.get('column_arrangement', 0):.6f}", f"{i.get('vector_calculation', 0):.6f}",
                        f"{i.get('matrix_vector_calc', 0):.6f}"
                    ])

    print(f"Benchmarking complete. Results saved to {output_file}")


def benchmark_freivalds(k, output_file="bench_freivalds_20.csv"):


    BASE_DIR = "../data_test_int"
    TYPES = ["band", "diagonal", "identity", "nilpotent", "ones",
             "permutation", "random", "random_max_value_n", "random_signed",
             "sparse_0.1", "sparse_signed", "symmetric", "toeplitz", "triangular",
             "vandermond_poly_n^{2}"

             ]
    US = ["u_0", "u_1"]
    RESULTS = []

    for matrix_type in TYPES:
        print(matrix_type)
        for u in US:
            dir_path = os.path.join(BASE_DIR, matrix_type, u)
            if not os.path.isdir(dir_path):
                continue

            files = os.listdir(dir_path)
            sizes = sorted({int(re.findall(r'\d+', f)[0]) for f in files if f.startswith("A_size_")})

            for size in sizes:
                print(size)
                A = torch.load(os.path.join(dir_path, f"A_size_{size}.pt")).to(dtype=torch.int64)
                B = torch.load(os.path.join(dir_path, f"B_size_{size}.pt")).to(dtype=torch.int64)
                C = torch.load(os.path.join(dir_path, f"C_size_{size}.pt")).to(dtype=torch.int64)

                start = time.perf_counter()
                result = freivalds(A, B, C, k)
                total_time = time.perf_counter() - start

                RESULTS.append({
                    "type": matrix_type,
                    "u": u,
                    "size": size,
                    "result": result,
                    "k": k,
                    "total_time": total_time
                })

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "u", "size", "k", "result", "total_time"])
        for r in RESULTS:
            writer.writerow([
                r["type"], r["u"], r["size"], r["k"], r["result"], f"{r['total_time']:.6f}"
            ])

    print(f"Benchmarking complete. Results saved to {output_file}")

def benchmark_approximation(b_fn, output_file="bench_approximation_n.csv"):



    BASE_DIR = "../data_test_int"
    TYPES = ["band", "diagonal", "identity", "nilpotent", "ones",
             "permutation", "random", "random_max_value_n", "random_signed",
             "sparse_0.1", "sparse_signed", "symmetric", "toeplitz", "triangular",
             "vandermond_poly_n^{2}"

             ]
    US = ["u_0", "u_1"]
    RESULTS = []

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "type", "u", "size", "b", "total_time",
            "abs_error", "max_error", "e1_norm", "fro_error",
            "sorting", "find_b_plus", "pos_sort", "merge"
        ])

        for matrix_type in TYPES:
            for u in US:
                dir_path = os.path.join(BASE_DIR, matrix_type, u)
                if not os.path.isdir(dir_path):
                    continue

                sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
                for size in sizes:
                    print(f"{matrix_type}, {u}, size={size}")
                    A = torch.load(os.path.join(dir_path, f"A_size_{size}.pt")).to(torch.float64)
                    B = torch.load(os.path.join(dir_path, f"B_size_{size}.pt")).to(torch.float64)
                    C_true = torch.load(os.path.join(dir_path, f"C_size_{size}.pt")).to(torch.float64)

                    b = b_fn(size)
                    print(f"b = {b}")

                    # Compute summary and approximation
                    S, total_time, timings = compute_summary(A, B, b)
                    C_hat = reconstruct_approximation_matrix(S, size)

                    # Sanity: non-zero count in approximation
                    print("nonzero entries in C_hat:", len(torch.nonzero(C_hat)))

                    # Compute error vs true product
                    abs_error = torch.abs(C_true - C_hat).sum().item()
                    max_error = torch.abs(C_true - C_hat).max().item()
                    e1_norm = C_true.abs().sum().item()
                    fro_error = torch.norm(C_true - C_hat).item()

                    # Lemma 6: max error should be ≤ e1_norm / b
                    error_bound_ratio = max_error / (e1_norm / b)

                    RESULTS.append({
                        "type": matrix_type,
                        "u": u,
                        "size": size,
                        "b": b,
                        "total_time": total_time,
                        "error_bound_ration": error_bound_ratio,
                        "abs_error": abs_error,
                        "max_error": max_error,
                        "e1_norm": e1_norm,
                        "fro_error": fro_error,
                        "timings": timings
                    })

                    t = timings
                    writer.writerow([
                        matrix_type, u, size, b, f"{total_time:.6f}",
                        f"{abs_error:.6f}", f"{max_error:.6f}",
                        f"{e1_norm:.6f}", f"{fro_error:.6f}",
                        f"{t.get('sorting', 0):.6f}",
                        f"{t.get('find_b_plus_largest_elements', 0):.6f}",
                        f"{t.get('positional_sort', 0):.6f}",
                        f"{t.get('merge', 0):.6f}"
                    ])

    print(f"Benchmarking complete. Results saved to {output_file}")


def benchmark_all_real(c, t_fn, primes=[], omegas=[], output_file="verification_real_world_t=n.csv"):


    BASE_DIR = "../matrices"  # points to your downloaded matrix dir
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "size", "t", "result", "total_time",
            "prime_calc", "omega_calc",
            "omega", "construct_poly", "FME", "compute_poly",
            "check_g", "internal_total", "iteration_total"
        ])

        for entry in os.listdir(BASE_DIR):
            full_path = os.path.join(BASE_DIR, entry)
            if not os.path.isdir(full_path):
                continue
            try:
                # Parse ID and size from folder name: id_{id}_{rows}x{cols}
                parts = entry.split("_")
                mid = int(parts[1])
                shape = parts[2]
                rows, cols = map(int, shape.split("x"))
                size = rows  # assuming square matrix

                # Load A and C (C = A @ A)
                A_data = torch.load(os.path.join(full_path, "A.pt"))
                C_data = torch.load(os.path.join(full_path, "A_matmul.pt"))

                A = A_data["tensor"].to_dense().to(dtype=torch.int64)
                C = C_data["tensor"].to_dense().to(dtype=torch.int64)
                B = A.t()

                t_val = t_fn(size)
                result, total_time, timing_data, timing_meta = verification(
                    A, B, C, c=c, t=t_val, primes=primes, omegas=omegas
                )

                m = timing_meta
                t = timing_data[0] if timing_data else {}

                writer.writerow([
                    mid, size, t_val, result, f"{total_time:.6f}",
                    f"{m.get('prime_calc', 0):.6f}", f"{m.get('omega_calc', 0):.6f}",
                    f"{t.get('omega', 0):.6f}", f"{t.get('construct_poly', 0):.6f}",
                    f"{t.get('FME', 0):.6f}", f"{t.get('compute_poly', 0):.6f}",
                    f"{t.get('check_g', 0):.6f}", f"{t.get('total', 0):.6f}",
                    f"{t.get('iteration_total', 0):.6f}"
                ])
                f.flush()
                print(f"✅ ID={mid}, size={size} → {total_time:.3f}s")

            except Exception as e:
                print(f"❌ Skipping {entry}: {e}")

    print(f"Benchmarking complete. Results saved to {output_file}")

def benchmark_matmul_real(output_file="bench_matmul_real.csv"):

    BASE_DIR = "../matrices"
    RESULTS = []

    for entry in os.listdir(BASE_DIR):
        dir_path = os.path.join(BASE_DIR, entry)
        if not os.path.isdir(dir_path):
            continue

        try:
            parts = entry.split("_")
            mid = int(parts[1])
            shape = parts[2]
            rows, cols = map(int, shape.split("x"))
            if rows != cols:
                print(f"❌ Skipped {entry}: non-square matrix {rows}x{cols}")
                continue
            size = rows

            # Load A and A_matmul
            A_data = torch.load(os.path.join(dir_path, "A.pt"))
            C_data = torch.load(os.path.join(dir_path, "A_matmul.pt"))

            A = A_data["tensor"].to_dense().to(dtype=torch.int64)
            B = A.t()
            C = C_data["tensor"].to_dense().to(dtype=torch.int64)

            start = time.perf_counter()
            result = torch.matmul(A, B)
            elapsed = time.perf_counter() - start

            RESULTS.append({
                "id": mid,
                "size": size,
                "result": True,  # You likely want a bool result, not the tensor
                "total_time": elapsed
            })
            print(f"✅ ID={mid} size={size} → {elapsed:.4f}s")

        except Exception as e:
            print(f"❌ Skipped {entry}: {e}")

    # Write CSV
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "size", "result", "total_time"])
        for r in RESULTS:
            writer.writerow([r["id"], r["size"], r["result"], f"{r['total_time']:.6f}"])

    print(f"Benchmarking complete. Results saved to {output_file}")


def benchmark_correct_speedUp_real(c, k_fn, output_file="bench_float_log=t.csv"):
    BASE_DIR = "../suit_sparse_real"
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "rows", "cols", "k", "total_time",
            "prime_calc", "first_iteration", "matrix_transposed",
            "second_iteration", "column_arrangement",
            "vector_calculation", "matrix_vector_calc", "check_g",
            "internal_total", "iteration_total"
        ])

        for shape_dir in os.listdir(BASE_DIR):
            shape_path = os.path.join(BASE_DIR, shape_dir)
            if not os.path.isdir(shape_path):
                continue

            for folder in os.listdir(shape_path):
                full_path = os.path.join(shape_path, folder)
                if not os.path.isdir(full_path):
                    continue

                try:
                    A_path = os.path.join(full_path, "A.pt")
                    B_path = os.path.join(full_path, "B.pt")
                    C_path = os.path.join(full_path, "Product.pt")



                    if not (os.path.exists(A_path) and os.path.exists(B_path) and os.path.exists(C_path)):
                        print(f"❌ Missing files in {folder}")
                        continue

                    A = torch.load(A_path)
                    B = torch.load(B_path)
                    C = torch.load(C_path)

                    if isinstance(A, torch.Tensor):
                        A = {"tensor": A}
                    if isinstance(B, torch.Tensor):
                        B = {"tensor": B}
                    if isinstance(C, torch.Tensor):
                        C = {"tensor": C}

                    A_tensor = A["tensor"].to(dtype=torch.float64)
                    B_tensor = B["tensor"].to(dtype=torch.float64)
                    C_tensor = C["tensor"].to(dtype=torch.float64)

                    test_copies = C_tensor.clone()
                    rows, cols = A_tensor.shape

                    if min(rows, cols) <= 2:
                        print(f"⚠️ Skipped {folder}: size too small ({rows}x{cols})")
                        continue
                    print(rows)
                    k_val = k_fn(rows)
                    C_error = generate_matrices.create_matrix_with_errors_torch(
                        C_tensor, k_val, rows, True).to(dtype=torch.float64)

                    result, total_time, timings, iteration_timings = correct_speedUp(
                        A_tensor, B_tensor, C_error, c=c, k=k_val, primes=[]
                    )

                    assert torch.equal(C_tensor, result)
                    # Write row
                    t = timings
                    i = iteration_timings
                    writer.writerow([
                        folder, rows, cols, k_val, f"{total_time:.6f}",
                        f"{t.get('prime_calculation', 0):.6f}",
                        f"{t.get('first_iteration', 0):.6f}",
                        f"{t.get('matrix_tranposed', 0):.6f}",
                        f"{t.get('second_iteration', 0):.6f}",
                        f"{i.get('column_arrangement', 0):.6f}",
                        f"{i.get('vector_calculation', 0):.6f}",
                        f"{i.get('matrix_vector_calc', 0):.6f}",
                        f"{i.get('check_g', 0):.6f}",
                        f"{i.get('internal_total', 0):.6f}",
                        f"{i.get('iteration_total', 0):.6f}"
                    ])
                except Exception as e:
                    print(f"❌ Skipped {folder}: {e}")

    print(f"✅ Benchmark complete. Output: {output_file}")


def cut_to_power_of_2(tensor: torch.Tensor) -> torch.Tensor:
    rows, cols = tensor.shape
    new_rows = 2 ** (rows.bit_length() - 1)
    new_cols = 2 ** (cols.bit_length() - 1)
    return tensor[:new_rows, :new_cols]

def benchmark_os_real(c, k_fn, output_file="bench_os_(n)real.csv"):



    BASE_DIR = "../matrices"
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "size", "k", "total_time",
            "prime_calc", "first_iteration",
            "matrix_transposed", "second_iteration",
            "column_arrangement", "vector_calculation",
            "matrix_vector_calc", "check_g",
            "internal_total", "iteration_total"
        ])

        for entry in os.listdir(BASE_DIR):
            full_path = os.path.join(BASE_DIR, entry)
            if not os.path.isdir(full_path):
                continue

            try:
                parts = entry.split("_")
                mid = int(parts[1])
                shape = parts[2]
                orig_rows, orig_cols = map(int, shape.split("x"))
                if orig_rows != orig_cols:
                    print(f"❌ Skipped {entry}: non-square matrix {orig_rows}x{orig_cols}")
                    continue
                if orig_rows > 850:
                    continue

                # Compute and save A_matmul.pt if missing
                if not os.path.exists(os.path.join(full_path, "A_matmul.pt")):
                    A = torch.load(os.path.join(full_path, "A.pt"))["tensor"]
                    if A.is_sparse:
                        A = A.to_dense()
                    A = A.to(dtype=torch.int64)
                    C = (A @ A).to(dtype=torch.int64)
                    torch.save({"tensor": C}, os.path.join(full_path, "A_matmul.pt"))

                # Load tensors
                A = torch.load(os.path.join(full_path, "A.pt"))["tensor"]
                if A.is_sparse:
                    A = A.to_dense()
                A = A.to(dtype=torch.int64)

                C = torch.load(os.path.join(full_path, "A_matmul.pt"))["tensor"]
                if C.is_sparse:
                    C = C.to_dense()
                C = C.to(dtype=torch.int64)


                A = cut_to_power_of_2(A)
                B = A.T
                C = torch.matmul(A, B)
                size = A.shape[0]


                k_val = k_fn(size)
                print(k_val)
                max_value = size

                C_error = generate_matrices.create_matrix_with_errors_torch(C, k_val, max_value, True)
                if C_error.is_sparse:
                    C_error = C_error.to_dense()
                C_error = C_error.to(dtype=torch.int64)
                t_sol = C - C_error

                A_n, B_n = change_to_verification_form_torch(A, B, C_error)


                primes = calculate_primes(size + 1, 1)
                if not primes:
                    continue
                prime = primes[0].item()

                result_matrix, total_time, timing_data = os_matrix_multiplication_mod_p(
                    A_n, B_n, k_val, prime
                )

                C_sol = result_matrix[:, -size:]
                print(size)
                print(total_time)
                assert torch.equal(C_sol, t_sol)

                writer.writerow([
                    mid, size, k_val, total_time,
                    timing_data.get("prime_calc", 0),
                    timing_data.get("first_iteration", 0),
                    timing_data.get("matrix_transposed", 0),
                    timing_data.get("second_iteration", 0),
                    timing_data.get("column_arrangement", 0),
                    timing_data.get("vector_calculation", 0),
                    timing_data.get("matrix_vector_calc", 0),
                    timing_data.get("check_g", 0),
                    timing_data.get("internal_total", 0),
                    timing_data.get("iteration_total", 0)
                ])

            except Exception as e:
                print(f"❌ Skipped {entry}: {e}")

    print(f"Benchmarking complete. Results saved to {output_file}")


def benchmark_naive(output_file="bench_naive.csv"):


    BASE_DIR = "../data_test_int"
    TYPES = [
             "random", "random", "random",
        "random"
    ]
    US = ["u_0", "u_1"]
    RESULTS = []

    for matrix_type in TYPES:
        print(matrix_type)
        for u in US:
            dir_path = os.path.join(BASE_DIR, matrix_type, u)
            if not os.path.isdir(dir_path):
                continue

            files = os.listdir(dir_path)
            #sizes = sorted({int(re.findall(r'\d+', f)[0]) for f in files if f.startswith("A_size_")})
            sizes = [8, 16, 32, 64, 128, 256, 512]
            for size in sizes:
                print(size)
                A = torch.load(os.path.join(dir_path, f"A_size_{size}.pt")).to(dtype=torch.int64)
                B = torch.load(os.path.join(dir_path, f"B_size_{size}.pt")).to(dtype=torch.int64)
                C = torch.load(os.path.join(dir_path, f"C_size_{size}.pt")).to(dtype=torch.int64)

                start = time.perf_counter()
                result = naive_matrix_multiplication(A, B)
                total_time = time.perf_counter() - start
                print(total_time)
                RESULTS.append({
                    "type": matrix_type,
                    "u": u,
                    "size": size,
                    "result": result,
                    "total_time": total_time
                })
                assert torch.equal(C, result)
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "u", "size", "k", "result", "total_time"])
        for r in RESULTS:
            writer.writerow([
                r["type"], r["u"], r["size"], r["k"], r["result"], f"{r['total_time']:.6f}"
            ])

    print(f"Benchmarking complete. Results saved to {output_file}")