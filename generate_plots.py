

def plot_calculated_results():
    import matplotlib.pyplot as plt

    # Matrix sizes
    matrix_sizes = [1000, 2000, 3000, 4000, 5000, 6000]

    # Data for verification with different t values
    verification_t2 = [0.4503195285797119, 1.9583184719085693, 4.198751926422119, 7.302755951881409,
                       11.306620717048645, 15.133582830429077]
    verification_log_n = [0.5844650268554688, 1.9355782270431519, 4.511630535125732, 8.348076343536377,
                          12.989210367202759, 19.266492247581482]
    verification_sqrt_n = [0.9115495681762695, 4.154887437820435, 10.921999335289001, 25.300981521606445,
                           32.595616698265076, 55.28094518184662]

    # Matrix multiplication times for comparison
    matrix_mult = [0.14043748378753662, 1.1609388589859009, 3.9197235107421875, 9.134621977806091,
                   17.626078963279724, 30.318474292755127]

    # Plotting
    plt.figure(figsize=(12, 8))

    # Plot verification times for each t value
    plt.plot(matrix_sizes, verification_t2, marker='o', label='Verification (t = 2)')
    plt.plot(matrix_sizes, verification_log_n, marker='o', label='Verification (t = log(n))')
    plt.plot(matrix_sizes, verification_sqrt_n, marker='o', label='Verification (t = sqrt(n))')


    # Plot matrix multiplication times for comparison
    plt.plot(matrix_sizes, matrix_mult, marker='o', linestyle='--', label='Matrix Multiplication')

    # Add labels, title, legend, and grid
    plt.xlabel("Matrix Size", fontsize=14)
    plt.ylabel("Running Time (seconds)", fontsize=14)
    plt.title("Verification Times for c = 0, max_value = n/100", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Save the plot
    plt.savefig("verification_c0_n100.png", dpi=300, bbox_inches='tight')

    # Show the plot
    plt.tight_layout()
    plt.show()