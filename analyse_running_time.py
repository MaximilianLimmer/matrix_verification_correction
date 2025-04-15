import time
import matplotlib.pyplot as plt

import numpy as np
from matplotlib.ticker import LogLocator, LogFormatter

import generate_matrices

from mm_verification import verification_torch


def measure_time(func, *args, **kwargs):

    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time

def mutliple_measure_time(list, func, *args, **kwargs):
    output = []
    for n in list:
        print(n)
        A, B = generate_matrices.generate_matrices_torch(n, n, n)
        C = verification_torch(A, B)
        time2, result2 = measure_time(func, A, B, C, 2,2, *args, **kwargs)
        output.append(time2)
    return output


def plot(function1, function2, output_filename,
         base_args1=(), kwargs1={}, args2=(), kwargs2={},
         input_sizes=range(1000, 5000, 500)):

    times_function1 = []
    times_function2 = []
    output = []

    for n in input_sizes:
        print(f"Processing input size: {n}")
        c = 1
        #A = generate_matrices.create_dense_matrix_with_nonzeros(n, n, 4 *  n, (1, n ** c))
        #B = generate_matrices.create_dense_matrix_with_nonzeros(n, n, 4 * n, (1, n ** c))
        #A, B = generate_matrices.generate_sparse_matrices_row_columns(n ,0, int(n/100), 0, int(n/100), n**c)
        A = generate_matrices.generate_random_diagonal_matrix(n, n ** c)
        B = generate_matrices.generate_random_diagonal_matrix(n, n ** c)
        #A, B = generate_matrices.generate_matrices_torch(n, n, n**c)


        print("klasdf")
        As, Bs = generate_matrices.generate_matrices_torch(n, n, n ** c)
        #Cs = verification_torch(As, Bs)
        #C = verification_torch(A, B)
        print("sdf")

        # Measure time for function2
        time2, result2 = measure_time(function2, A, B, *args2, **kwargs2)
        times_function2.append(time2)

        print("-------------------------------------------------------------")
        #C_error = generate_matrices.create_matrix_with_errors_torch(result2, 1)

        # Measure time for function1
        args1 = base_args1
        time1, result1 = measure_time(function1, A, B, result2, c, 2, *args1, **kwargs1)
        times_function1.append(time1)

        output.append((result1))
        print(result1)

        # Plot the results with x-axis in log scale and y-axis in log scale
    plt.figure(figsize=(10, 6))
    plt.plot(input_sizes, times_function1, label="verification", marker='o')
    plt.plot(input_sizes, times_function2, label="matrix multiplication", marker='x')

    # Set x-axis and y-axis to natural log scale
    plt.xscale('log', base=np.e)  # Natural log scale for x-axis
    plt.yscale('log', base=np.e)  # Natural log scale for y-axis

    # Configure natural log tick formatting for x-axis
    loc_x = LogLocator(base=np.e)  # Set base to e for x-axis
    fmt_x = LogFormatter(base=np.e, labelOnlyBase=False)  # Format with e-base labels for x-axis

    plt.gca().xaxis.set_major_locator(loc_x)
    plt.gca().xaxis.set_major_formatter(fmt_x)

    # Configure natural log tick formatting for y-axis
    loc_y = LogLocator(base=np.e)  # Set base to e for y-axis
    fmt_y = LogFormatter(base=np.e, labelOnlyBase=False)  # Format with e-base labels for y-axis

    plt.gca().yaxis.set_major_locator(loc_y)
    plt.gca().yaxis.set_major_formatter(fmt_y)

    plt.xlabel("Log(Input Size) (ln(n))")
    plt.ylabel("Log(Running Time) (ln(seconds))")
    plt.title("Running Time Comparison (Natural Log Scales)")

    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(output_filename)
    plt.show()
    return output


def plot_function_values(*functions):
    plt.figure(figsize=(10, 6))

    for func_name, values in functions:
        x = np.linspace(1000, 1000 * len(values), len(values))  # Create x-axis values (1000, 2000, ...)
        plt.plot(x, values, marker='o', label=f'{func_name}')

    plt.xlabel('x values')
    plt.ylabel('Function values')
    plt.title('Plot of Functions')
    plt.legend()
    plt.grid(True)
    plt.show()