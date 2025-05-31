import gc
import math

import matplotlib.pyplot as plt

import torch
import os
from typing import Callable, List, Any

from correction_gas import correction_algorithmica
import generate_matrices
from bench import analyse_running_time
from correction_gas.correction_algorithmica import correct_speedUp, primes_correction

from fme_correct_verify.mm_verification import calculate_primes, calculate_primitive_roots, verification_torch


def generate_and_save(
    output_dir: str,
    sizes: List[int],
    count_per_size: int,
    creation_fn: Callable[..., Any],
    file_name_template: str,
    *args
):

    os.makedirs(output_dir, exist_ok=True)

    for size in sizes:
        objects = [creation_fn(size, size, *args) for _ in range(count_per_size)]
        max_value = max(obj.max().item() if isinstance(obj, torch.Tensor) else max(obj) for obj in objects)
        file_path = os.path.join(output_dir, f"tensors_{size}_max{size}.pt")
        torch.save(objects, file_path)
        print(f"Saved {count_per_size} primitive roots for size {size} to {file_path}")

    print("Dataset generation and saving complete!")

def load_primes(input_dir: str, size: int, d: int) -> Any:

    # Construct the expected file name
    file_name = f"primes_{size}_d_{d}.pt"
    file_path = os.path.join(input_dir, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_name} not found in directory {input_dir}")

    print(f"Loading primes from {file_path}")
    primes = torch.load(file_path)
    return primes


def load_tensors(input_dir: str, file_name: str,  size: int):

    # Find the file matching the size pattern
    file_name_pattern = file_name
    files = [f for f in os.listdir(input_dir) if file_name_pattern in f]

    if not files:
        raise FileNotFoundError(f"No file found for tensors of size {size} in {input_dir}")

    file_path = os.path.join(input_dir, files[0])  # Assuming one file per size
    print(f"Loading tensors from {file_path}")

    # Load the tensors
    tensors = torch.load(file_path)
    return tensors

def load_primitive_roots(input_dir: str, size: int) -> Any:

    # Construct the expected file name
    file_name = f"primitive_roots_for_primes_of{size}.pt"
    file_path = os.path.join(input_dir, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_name} not found in directory {input_dir}")

    print(f"Loading primitive roots from {file_path}")
    primitive_roots = torch.load(file_path)
    return primitive_roots

def calculate_tensor_solutions_pair(input_dir: str, output_dir: str, size: int, pair_index: int):
    """
    Calculate and save a single pair of tensor solutions incrementally.

    Parameters:
        input_dir (str): Directory where tensors are stored.
        output_dir (str): Directory where solutions will be saved.
        size (int): Size of the tensors to process.
        pair_index (int): Index of the tensor pair to calculate (e.g., 0 for 0 and 10, 1 for 1 and 11).
    """
    os.makedirs(output_dir, exist_ok=True)

    tensor_file = os.path.join(input_dir, f"tensors_{size}_max{size}.pt")
    if not os.path.exists(tensor_file):
        raise FileNotFoundError(f"Tensor file {tensor_file} not found.")

    print(f"Loading tensors from {tensor_file}")
    tensors = torch.load(tensor_file)
    print(len(tensors))

    if pair_index + 5 >= len(tensors):
        raise IndexError(f"Pair index {pair_index} out of range for tensors of size {size}.")

    print(f"Calculating product of tensors {pair_index} and {pair_index + 5} for size {size}")
    solution = torch.matmul(tensors[pair_index], tensors[pair_index + 5])

    solution_file = os.path.join(output_dir, f"solutions_{size}.pt")

    # Load existing solutions if the file already exists
    if os.path.exists(solution_file):
        existing_solutions = torch.load(solution_file)
    else:
        existing_solutions = []

    existing_solutions.append(solution)

    # Save the updated solutions
    torch.save(existing_solutions, solution_file)
    print(f"Saved solution for pair {pair_index} of size {size} to {solution_file}")
    return

# Usage for primitive roots:
def primitive_roots_creation_fn(size, d):
    primes = calculate_primes(size, d)
    return calculate_primitive_roots(primes)




def benchmarks_verification(function1, sizes, d, c, t, precalculated_primes_roots):

    list = []
    primes = []
    primitive_roots = []

    for x in sizes:
        time = 0

        if precalculated_primes_roots:
            primes = load_primes("primes_dataset", x, d)
            primitive_roots = load_primitive_roots("primitive_roots_dataset", x)

        solutions = load_tensors("solutions_dataset", f"solutions_{x}", x)
        tensors = load_tensors("tensors_dataset", f"tensors_{x}", x)


        amount_of_tensors = len(tensors)
        amount_of_runs = int(amount_of_tensors / 2)


        for i in range(2):

            time += analyse_running_time.measure_time(function1, tensors[i], tensors[i + amount_of_runs],
                                                      solutions[i], c, int(math.sqrt(x)), primes[0], primitive_roots[0])
            print(time)
            gc.collect()

        average_time = time / 2
        list.append(average_time)
        del tensors
        del solutions
        del primes
        del primitive_roots


    return list


def benchmarks_verification_change_max_value(function1, sizes, d, c, t, precalculated_primes_roots, max_value_multiplier):

    list = []
    primes = []
    primitive_roots = []

    for x in sizes:
        time = 0

        if precalculated_primes_roots:
            primes = load_primes("primes_dataset", x, d)
            primitive_roots = load_primitive_roots("primitive_roots_dataset", x)

        solutions = load_tensors("solutions_dataset", f"solutions_{x}", x)
        tensors = load_tensors("tensors_dataset", f"tensors_{x}", x)


        amount_of_tensors = len(tensors)
        amount_of_runs = int(amount_of_tensors / 2)


        for i in range(2):
            A = tensors[i] * max_value_multiplier
            A = A.to(torch.int64)
            B = tensors[i + amount_of_runs] * max_value_multiplier
            B = B.to(torch.int64)
            C = verification_torch(A, B)
            C = C.to(torch.int64)

            time += analyse_running_time.measure_time(function1, A, B,
                                                      C, c - 1, int(math.sqrt(x)), primes[0], primitive_roots[0])
            print(time)
            gc.collect()

        average_time = time / 2
        list.append(average_time)
        del tensors
        del solutions
        del primes
        del primitive_roots


    return list

def benchmark_matrix_multiplication(sizes):
    output = []
    for x in sizes:
        #tensors = load_tensors("tensors_dataset", f"tensors_{x}", x)

        #amount_of_tensors = len(tensors)
        #amount_of_runs = int(amount_of_tensors / 2)

        time = 0

        for i in range(2):
            #A, B = generate_pair_diagonal_matrices_torch(x, x, x)
            A = random_band_matrix(x, x, 1, 1)
            B = random_band_matrix(x, x, 1, 1)

            time += analyse_running_time.measure_time(verification_torch, A, B)
            print(time)
            gc.collect()

        average_time = time / 2
        output.append(average_time)
    return output


def benchmarks_verification_diagonal_matrices(function1, sizes, d, c, t, precalculated_primes_roots, amount_of_runs):

    list = []
    primes = []
    primitive_roots = []

    for x in sizes:
        time = 0

        if precalculated_primes_roots:
            primes = load_primes("primes_dataset", x, d)
            primitive_roots = load_primitive_roots("primitive_roots_dataset", x)


        for i in range(amount_of_runs):
            A, B = generate_pair_diagonal_matrices_torch(x, x, x)
            C = verification_torch(A, B)
            time += analyse_running_time.measure_time(function1, A, B,
                                                      C, c, int(math.sqrt(x)), primes[0], primitive_roots[0])
            print(time)
            gc.collect()

        average_time = time / amount_of_runs
        list.append(average_time)

        del primes
        del primitive_roots


    return list

def benchmarks_verification_band_matrices(function1, sizes, d, c, t, precalculated_primes_roots, amount_of_runs):

    list = []
    primes = []
    primitive_roots = []

    for x in sizes:
        time = 0

        if precalculated_primes_roots:
            primes = load_primes("primes_dataset", x, d)
            primitive_roots = load_primitive_roots("primitive_roots_dataset", x)


        for i in range(amount_of_runs):
            print(x)
            A = random_band_matrix(x, x, 1, 1)
            B = random_band_matrix(x, x, 1, 1)
            C = verification_torch(A, B)
            time += analyse_running_time.measure_time(function1, A, B,
                                                      C, c, int(math.sqrt(x)), primes[0], primitive_roots[0])
            print(time)
            gc.collect()

        average_time = time / amount_of_runs
        list.append(average_time)

        del primes
        del primitive_roots


    return list

def correction_c_test(size, c, k, amount_of_runs, load):
    #primes = primes_correction(c, math.sqrt(k), size)
    primes =  [47, 53]
    counter = 0
    if load:
        print("load")
        solutions = load_tensors("solutions_dataset", f"solutions_{size}", size)
        tensors = load_tensors("tensors_dataset", f"tensors_{size}", size)

    for i in range(amount_of_runs):
        print(i)
        if load:
            A = tensors[i]
            B = tensors[i + amount_of_runs]
            C = solutions[i]
        else:
            A, B = generate_matrices_torch(size, size, size)
            C = torch.matmul(A, B)
        errors = 0
        C_error = generate_matrices.create_matrix_with_errors_torch(C, 1, 1000)
        C_input = C_error.clone()
        difference = abs(torch.sub(C, C_error))
        print(difference.max())
        print(primes)
        C_out = correction_algorithmica.correct_speedUp(A, B, C_input, c, k, primes)
        n, m = C.shape
        print(m,n)
        print(C_out)
        assert torch.equal(C_out, C)

    return counter / amount_of_runs

def correction_benchmark(sizes, c, k, amount_of_runs):
    times = []
    for x in sizes:
        time = 0

        for i in range(amount_of_runs):
            print(i)
            A, B = generate_matrices_torch(x, x, x)
            C = verification_torch(A, B)
            primes = primes_correction(c, math.sqrt(k), x)
            print(primes)
            C_error = generate_matrices.create_matrix_with_errors_torch(C, 10000)

            time += analyse_running_time.measure_time(correct_speedUp, A, B, C_error, c, k, primes)
            print(time)
        average_time = time / amount_of_runs
        times.append(average_time)
    return times

def plot_function_results(data, x_values, titel):

    if not all(len(x_values) == len(results[0]) for _, results in data):
        raise ValueError("All results must have the same length as x_values.")

    plt.figure(figsize=(10, 6))

    for name, results in data:
        for i, result in enumerate(results):
            plt.plot(x_values, result, label=f"{name} - Run {i + 1}")

    plt.xlabel("Matrix size")
    plt.ylabel("Running Time in seconds")
    plt.title(titel)
    plt.legend()
    plt.grid(True)
    plt.show()

