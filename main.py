import torch

from bench.bench_new import benchmark_correct_speedUp

from bench import bench_new
import math

from generate_matrices import create_file_structure_with_rank1_support


def verification_run_bench():
    t_fn_const = lambda size: 1
    t_fn_double = lambda size: size * 2
    t_fn_log = lambda size: max(1, int(math.log2(size)))
    t_fn_sqrt = lambda size: int(math.sqrt(size))
    t_fn_n = lambda size: size

    # Example run:
    bench_new.benchmark_all(c=1, t_fn=t_fn_log)

def correction_run_bench():
    import torch

    A = torch.load("data_test_int/ones/u_0/A_size_32.pt")  # Replace XX with actual size

    c = 1.3841

    t_fn_const = lambda size: 4
    t_fn_double = lambda size: size * 2
    t_fn_log = lambda size: max(1, int(math.log2(size)))
    t_fn_sqrt = lambda size: int(math.sqrt(size))
    t_fn_n = lambda size: size // 2

    bench_new.benchmark_correct_speedUp_real(c, t_fn_log)

def os_run_bench():
    t_fn_const = lambda size: 4
    t_fn_double = lambda size: size * 2
    t_fn_log = lambda size: max(1, int(math.log2(size)))
    t_fn_sqrt = lambda size: int(math.sqrt(size))
    t_fn_n = lambda size: size

    def d(n):
        return n

    # Example run:
    bench_new.benchmark_os_real(1, d)


def os_corr_run_bench():
    t_fn_const = lambda size: 2
    t_fn_double = lambda size: size * 2
    t_fn_log = lambda size: max(1, int(math.log2(size)))
    t_fn_sqrt = lambda size: int(math.sqrt(size))
    t_fn_n = lambda size: size // 2

    # Example run:
    bench_new.benchmark_all_os_mm_as_correction(c=1, t_fn=t_fn_n)

def generate_sparse_n_squared():
    sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    for size in sizes:
        create_file_structure_with_rank1_support(
            matrix_types = ["random_max_value_n", "random_max_value_n^2",
                            "sparse_0.1_max_value_n^2", "sparse_0.1_max_value_size",
                            "vandermonde_poly_n^{2}"],
            sizes=[size],
            max_value=size,  # Larger range: max_value = n²
            dtype=torch.int64,
            sparsity=0.1
        )

def plot():
    import numpy as np
    import matplotlib.pyplot as plt

    # Matrix sizes
    sizes = np.array([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])

    # Triangle matrices
    t_mat_mul = np.array([0.00045775, 2.375e-05, 2.85e-05, 5.05e-05, 0.0001005, 0.00039325, 0.002782, 0.0205955, 0.16739075, 1.2447775])
    t_random_n = np.array([0.00052975, 0.00119775, 0.00595575, 0.05309125, 0.1540185, 0.79762075, 3.703629, 16.62639175, 82.1546185, np.nan])

    # Filter to sizes >= 16
    mask = sizes >= 16
    sizes = sizes[mask]
    t_mat_mul = t_mat_mul[mask]
    t_random_n = t_random_n[mask]

    # ----- Plot -----
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, t_mat_mul, marker='o', label='torch')
    plt.plot(sizes, t_random_n, marker='s', label='Verification for t = n')

    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.xlabel('Matrix size (n)', fontsize=12)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    plt.title('Runtime: Comparison torch vs Verification', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("torch_verification.pdf", dpi=300)
    plt.show()




def plot_correction():
    import numpy as np
    import matplotlib.pyplot as plt

    # Matrix sizes
    sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    # Absolute errors (float)
    t_navie_matrix = np.array([0.004628, 0.033939, 0.269287, 2.076098, 16.557327, 132.259346, 1062.671445])

    t_correction_4_float = np.array([
        0.0086105, 0.0091965, 0.0145315, 0.025943, 0.0358115,
        0.056579, 0.154919, 0.4662655, 1.5010315, 6.5971255
    ])
    t_correction_log_float = np.array([
        0.0061605, 0.008861, 0.016739, 0.0308965, 0.057613,
        0.1091955, 0.250604, 1.0410635, 4.563249, 19.4874175
    ])
    t_correction_sqrt_float = np.array([
        0.0060215, 0.0089185, 0.016957, 0.051348, 0.08403,
        0.216651, 0.5558195, 3.4384385, 21.232577, 122.99876
    ])

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(sizes[:7], t_navie_matrix, marker='o', label='Naive MM')
    plt.plot(sizes, t_correction_4_float, marker='s', label='Correction t=4')
    plt.plot(sizes, t_correction_log_float, marker='^', label='Correction t=log(n)')
    plt.plot(sizes, t_correction_sqrt_float, marker='d', label='Correction t=sqrt(n)')

    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.xlabel('Matrix size (n)', fontsize=12)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    plt.title('Error Comparison Runtime (log₂): Naive MM vs Correction ', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("error_comparison_naive.pdf", dpi=300, bbox_inches='tight', transparent=True)
    plt.show()




    # ----- Plot 1: Triangle vs Random, log(n) and sqrt(n) -----


def plot_os_naive():
    import numpy as np
    import matplotlib.pyplot as plt

    sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    t_navie_matrix = np.array([0.004628, 0.033939, 0.269287, 2.076098, 16.557327, 132.259346, 1062.671445])
    t_os_n_int = np.array([0.0041665, 0.02090225, 0.1200295, 3.3004015, 11.71011975, 82.117683, 690.39305175])
    t_os_log_int = np.array([0.002164, 0.00615375, 0.018478, 0.067686, 1.5376975, 2.1463745, 6.6177315, 16.39132375, 66.76478975, 277.53756075])
    t_os_sqrt_int = np.array([0.00166575, 0.006372, 0.01999475, 0.09724575, 1.43659875, 5.05902275, 12.621258, 65.3288515, 343.749269])

    plt.figure(figsize=(10, 6))
    plt.plot(sizes[:len(t_navie_matrix)], t_navie_matrix, marker='o', label='Naive matrix mult.')
    plt.plot(sizes[:len(t_os_n_int)], t_os_n_int, marker='d', label='Correction t=n')
    #plt.plot(sizes[:len(t_os_log_int)], t_os_log_int, marker='s', label='Correction t=log(n)')
    #plt.plot(sizes[:len(t_os_sqrt_int)], t_os_sqrt_int, marker='^', label='Correction t=sqrt(n)')

    #plt.xscale('log', base=2)
    #plt.yscale('log', base=2)
    plt.xlabel('Matrix size (n)', fontsize=12)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    plt.title('OS Matrix Multiplication Runtime: Comparison for t=n and naive mm', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("os_runtime_naive.pdf", dpi=300, bbox_inches='tight', transparent=True)
    plt.show()


def os_plot_triangular_bar():
    import numpy as np
    import matplotlib.pyplot as plt

    sizes = np.array([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])

    t_os_log_int_tri = np.array([
        0.0015765, 0.0057155, 0.0192145, 0.0617355, 0.2102975,
        1.717942000000001, 2.8297025, 11.5884725, 45.808704, 183.7838435
    ])

    t_os_log_int_rand = np.array([
        0.002164, 0.00615375, 0.018478, 0.067686, 1.5376975,
        2.1463745, 6.6177315, 16.39132375, 66.76478975, 277.53756075
    ])

    bar_width = 0.35
    indices = np.arange(len(sizes))

    plt.figure(figsize=(12, 6))
    plt.bar(indices - bar_width / 2, t_os_log_int_rand, bar_width, label='Random', log=True)
    plt.bar(indices + bar_width / 2, t_os_log_int_tri, bar_width, label='Triangular', log=True)

    plt.xticks(indices, sizes)
    plt.xlabel('Matrix size (n)', fontsize=12)
    plt.ylabel('Runtime (seconds, log scale)', fontsize=12)
    plt.title('Runtime Comparison: Random vs Triangular Matrices (t=log(n))', fontsize=14)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("os_runtime_bar_comparison.pdf", dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

def regression_os():
    import numpy as np
    from scipy.stats import linregress

    sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    runtimes = np.array([0.00166575, 0.006372, 0.01999475, 0.09724575, 1.43659875, 5.05902275, 12.621258, 65.3288515, 343.749269])

    log_n = np.log2(sizes)
    log_t = np.log2(runtimes)

    slope, intercept, r_value, _, _ = linregress(log_n, log_t)

    print(f"Estimated exponent a ≈ {slope:.3f}")
    print(f"R² ≈ {r_value ** 2:.4f}")

def plot_verification_vs_freivald():
    import matplotlib.pyplot as plt

    # Runtime values
    freivalds_times = [0.000320, 0.000370, 0.000490, 0.001160, 0.003200, 0.013350, 0.066420, 0.353500, 2.440100,
                       10.556900]
    deterministic_times = [0.00047, 0.00049625, 0.00173375, 0.00549425, 0.01912825, 0.0703345, 0.301089, 1.1038095,
                           4.243267, 17.66258475]
    matrix_sizes = [2 ** i for i in range(3, 13)]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(matrix_sizes, freivalds_times, marker='x', label='Freivalds (k=20)')
    plt.plot(matrix_sizes, deterministic_times, marker='s', label='t = log(n)')

    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.xlabel('Matrix size (n)')
    plt.ylabel('Runtime (s)')
    plt.title('Runtime: Freivalds vs Deterministic')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save as high-quality PDF
    plt.savefig("freivalds_vs_deterministic.pdf", format='pdf', dpi=300)

    plt.show()

def plot_approx():
    import numpy as np
    import matplotlib.pyplot as plt

    # Matrix sizes
    sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    # Runtime data for different correction settings
    t_approx_log_int = np.array([0.000117, 0.000215, 0.000575, 0.001671, 0.005304, 0.027345, 0.094803, 0.374938, 1.574996, 6.674834]
)
    t_approx_sqrt_int = np.array([0.00011325, 0.00025, 0.00070125, 0.00205325, 0.006496, 0.03299825, 0.11127875, 0.438699, 1.73213775, 6.9796685]
)
    t_approx_n_int = np.array([0.0002105, 0.0006145, 0.002048, 0.0076795, 0.031603, 0.1337695, 0.560706, 2.5818515, 11.304386, 53.9271565]
)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(sizes[:len(t_approx_n_int)], t_approx_n_int, marker='d', label='Approx t=n')
    plt.plot(sizes[:len(t_approx_sqrt_int)], t_approx_sqrt_int, marker='s', label='Approx t=sqrt(n)')
    plt.plot(sizes[:len(t_approx_log_int)], t_approx_log_int, marker='^', label='Approx t=log(n)')

    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.xlabel('Matrix size (n)', fontsize=12)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    plt.title('Approx Matrix Multiplication Runtime (log₂): Influence of number of approximated elements)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("approx_runtime.pdf", dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

def scatter_plot():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.cm as cm
    import os

    csv_files = [
        ("bench_float_t=4.csv", "t = 4"),
        ("bench_float_log=t.csv", "t = log(n)"),
        ("bench_float_sqrt=t.csv", "t = sqrt(n)")
    ]

    all_data = []

    for file, label in csv_files:
        if not os.path.exists(file):
            print(f"❌ Missing file: {file}")
            continue

        df = pd.read_csv(file)
        if "rows" not in df.columns or "total_time" not in df.columns:
            print(f"⚠️ Skipping {file} (missing required columns)")
            continue

        df = df[["rows", "total_time"]].copy()
        df["rows"] = pd.to_numeric(df["rows"], errors="coerce")
        df["total_time"] = pd.to_numeric(df["total_time"], errors="coerce")
        df.dropna(subset=["rows", "total_time"], inplace=True)
        df = df[(df["rows"] >= 50) & (df["total_time"] > 0)]
        df["source"] = label
        all_data.append(df)

    if not all_data:
        raise ValueError("No usable data files found.")

    df_all = pd.concat(all_data, ignore_index=True)

    colors = cm.tab10.colors
    plt.figure(figsize=(8, 6))

    for i, (label, group) in enumerate(df_all.groupby("source")):
        plt.scatter(
            np.log2(group["rows"]),
            np.log2(group["total_time"]),
            alpha=0.7,
            label=label,
            color=colors[i % len(colors)]
        )

    def format_tick(x, _):
        return f"$2^{{{int(x)}}}$"

    xticks_log2 = sorted(set(int(np.log2(s)) for s in df_all["rows"]))
    yticks_log2 = sorted(set(int(np.log2(t)) for t in df_all["total_time"]))

    plt.xticks(xticks_log2, [format_tick(x, None) for x in xticks_log2])
    plt.yticks(yticks_log2, [format_tick(y, None) for y in yticks_log2])

    plt.xlabel("Size (rows)")
    plt.ylabel("Total Time")
    plt.title("Real World Matrices – Correction Runtime")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("scatter_float_real.pdf", format="pdf", dpi=300)
    print("✅ Saved plot to scatter_float_real.pdf")


def plot_comparison():
    import numpy as np
    import matplotlib.pyplot as plt

    t_os_sqrt_int = np.array([0.00166575, 0.006372, 0.01999475, 0.09724575, 1.43659875, 5.05902275, 12.621258, 65.3288515, 343.749269])

    total_times_new = [0.394981, 3.714714, 19.496037, 132.480280, 681.06]
    t_correction_sqrt_float = np.array([
        0.0060215, 0.0089185, 0.016957, 0.051348, 0.08403,
        0.216651, 0.5558195, 3.4384385, 21.232577, 122.99876
    ])
    matrix_sizes = [2 ** i for i in range(3, 13)]

    plt.figure(figsize=(10, 6))
    plt.plot(matrix_sizes[:len(t_os_sqrt_int)], t_os_sqrt_int, marker='x', label='Künnemann os-MM')
    plt.plot(matrix_sizes[:len(total_times_new)], total_times_new, marker='s', label='Kutzkov Approximation')
    plt.plot(matrix_sizes[:len(t_correction_sqrt_float)], t_correction_sqrt_float, marker='o', label='Gąsieniec Correction')

    #plt.xscale('log', base=2)
    #plt.yscale('log', base=2)
    plt.xlabel('Matrix size (n)')
    plt.ylabel('Runtime (s)')
    plt.title('Runtime: Comparison of all correction approaches for t = sqrt(n)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("all_comparison_sqrt(n).pdf", format='pdf', dpi=300)

    plt.show()

if __name__ == "__main__":

    #plot_comparison()

    def d(n):
        return int(math.sqrt(n))

    benchmark_correct_speedUp(1, d)


