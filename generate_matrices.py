import random
import torch
import os
from scipy.io import mmread
from typing import Tuple

from fme_correct_verify import mm_verification


def load_and_save_as_torch(matrix_path, save_path, dtype=torch.float32):
    """
    Load a matrix from a .mtx file, convert it to a dense PyTorch tensor, and save it.

    Args:
        matrix_path (str): Path to the .mtx file.
        save_path (str): Path to save the PyTorch tensor.
        dtype (torch.dtype): Data type for the PyTorch tensor (default: torch.float32).

    Returns:
        torch.Tensor: The loaded and converted PyTorch tensor.
    """
    # Step 1: Load the matrix from the .mtx file
    matrix = mmread(matrix_path)

    # Step 2: Convert the sparse_0.1 matrix to a dense NumPy array
    dense_matrix = matrix.toarray()

    # Step 3: Convert the NumPy array to a PyTorch tensor
    tensor = torch.tensor(dense_matrix, dtype=dtype)

    # Step 4: Save the tensor to the specified path
    torch.save(tensor, save_path)

    print(f"Matrix successfully loaded, converted, and saved to {save_path}")
    return tensor

def create_file_structure_with_rank1_support(
    matrix_types,
    sizes,
    max_value,
    dtype=torch.float32,  # Enforced default to float
    sparsity=0.1,
    nonzero_fn_map=None
):
    """
    Generates matrices of specified types and sizes with float dtype.
    """
    if not os.path.exists("data_test_float"):
        os.makedirs("data_test_float")

    for matrix_type in matrix_types:
        print(f"Generating {matrix_type} matrices...")
        for size in sizes:
            print(f"  Size {size}")
            for u in range(2):  # Two instances per size
                instance_dir = os.path.join("data_test_int", matrix_type, f"u_{u}")
                os.makedirs(instance_dir, exist_ok=True)

                # Rank-1 controlled case
                if matrix_type.startswith("rank1_"):
                    nonzero_fn = (
                        nonzero_fn_map.get(matrix_type)
                        if nonzero_fn_map and matrix_type in nonzero_fn_map
                        else lambda n: int(sparsity * n)
                    )
                    A, B, C = generate_rank1_product_controlled(
                        n=size,
                        max_value=max_value,
                        dtype=dtype,
                        nonzero_fn=nonzero_fn
                    )
                else:
                    A, B, C = generate_pair_solution_matrices(
                        n=size,
                        l=size,
                        max_value=max_value,
                        dtype=dtype,
                        matrix_type=matrix_type,
                        sparsity=sparsity
                    )

                # Save
                save_matrices(
                    A, B, C,
                    save_dir=instance_dir,
                    name_A=f"A_size_{size}.pt",
                    name_B=f"B_size_{size}.pt",
                    name_C=f"C_size_{size}.pt"
                )


def generate_and_save_matrices(
    n: int,
    l: int,
    max_value: int,
    dtype: torch.dtype,
    matrix_type: str,
    sparsity: float,
    save_dir: str,
    name_A: str,
    name_B: str,
    name_C: str
):
    if matrix_type == "rank1_sqrt(n)_nonzeroes":
        nonzero_fn = lambda n: int(sparsity * n)  # or any other logic you want
        A, B, C = generate_rank1_product_controlled(n, max_value, dtype, nonzero_fn)
    else:
        A, B, C = generate_pair_solution_matrices(n, l, max_value, dtype, matrix_type, sparsity)

    save_matrices(A, B, C, save_dir, name_A, name_B, name_C)


def save_matrices(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    save_dir: str,
    name_A: str,
    name_B: str,
    name_C: str
):
    """
    Saves matrices A, B, and C to disk.

    Args:
        A (torch.Tensor): First matrix.
        B (torch.Tensor): Second matrix.
        C (torch.Tensor): Product matrix.
        save_dir (str): Directory to save the matrices.
        name_A (str) name of the matrix A
        name_B (str) name of the matrix B
        name_C (str) name of the matrix C
    """
    os.makedirs(save_dir, exist_ok=True)

    torch.save(A, os.path.join(save_dir, name_A))
    torch.save(B, os.path.join(save_dir, name_B))
    torch.save(C, os.path.join(save_dir, name_C))


def generate_rank1_product_controlled(
    n: int,
    max_value: int,
    dtype: torch.dtype,
    nonzero_fn,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates matrices A and B such that AB has exactly t = nonzero_fn(n) nonzero entries,
    each created via independent rank-1 contributions.

    Args:
        n (int): Size of square matrices.
        max_value (int): Maximum value for entries in A and B.
        dtype (torch.dtype): Desired tensor type.
        nonzero_fn (Callable[[int], int]): Function returning desired number of nonzeros in AB.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A, B, and AB = C.
    """
    t = min(nonzero_fn(n), n * n)
    A = torch.zeros((n, n), dtype=dtype)
    B = torch.zeros((n, n), dtype=dtype)
    C = torch.zeros((n, n), dtype=dtype)

    used_positions = set()

    for _ in range(t):
        while True:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if (i, j) not in used_positions:
                used_positions.add((i, j))
                break

        k = random.randint(0, n - 1)
        a = random.randint(1, max_value)
        b = random.randint(1, max_value)

        A[i, k] = a
        B[k, j] = b
        C[i, j] = a * b  # Since we know exactly what's being added

    return A, B, C


def existing_solution_and_errors(
    A: torch.Tensor,
    B: torch.Tensor,
    absolute_error: int,
    positive_error: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """

    :param A: first matrix
    :param B: second matrix
    :param absolute_error: number of generated errors in the erroneous version
    :param positive_error: determines if the error can only be a positive value
    :return: the correct matrix product and the erroneous version
    """
    C = torch.matmul(A, B)
    max_abs_element = int(C.abs().max() * C.sign()[C.abs().argmax()])
    C_error = create_matrix_with_errors_torch(C, absolute_error, max_abs_element, positive_error)
    return C, C_error

def solutions_for_existing(
    A: torch.Tensor,
    B: torch.Tensor
) -> torch.Tensor:
    """
    Calculate a solution for an already existing pair of matrices
    :param A: first matrix
    :param B: second matrix
    :return: calculated matrix
    """
    return torch.matmul(A, B)


def generate_pair_matrices(
    n: int,
    l: int,
    max_value: int,
    dtype: torch.dtype,
    matrix_type: str,
    sparsity: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a pair of matrices of the specified type.

    Args:
        n (int): Number of rows for matrix A and columns for matrix B.
        l (int): Number of columns for matrix A and rows for matrix B.
        max_value (int): Maximum value for the matrix elements.
        dtype (torch.dtype): Data type of the matrices.
        matrix_type (str): Type of matrix to generate. Options: 'random', 'sparse_0.1', 'toeplitz', 'diagonal', 'gaussian'.
        sparsity (float): Sparsity level for sparse_0.1 matrices (default: 0.1).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Generated matrices A and B.
    """
    A = generate_matrix(n, l, max_value, dtype, matrix_type, sparsity)
    B = generate_matrix(l, n, max_value, dtype, matrix_type, sparsity)
    return A, B

def generate_pair_solution_matrices(
    n: int,
    l: int,
    max_value: int,
    dtype: torch.dtype,
    matrix_type: str,
    sparsity: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates a pair of matrices and their product.

    Args:
        n (int): Number of rows for matrix A and columns for matrix B.
        l (int): Number of columns for matrix A and rows for matrix B.
        max_value (int): Maximum value for the matrix elements.
        dtype (torch.dtype): Data type of the matrices.
        matrix_type (str): Type of matrix to generate. Options: 'random', 'sparse_0.1', 'toeplitz', 'diagonal', 'gaussian'.
        sparsity (float): Sparsity level for sparse_0.1 matrices (default: 0.1).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Generated matrices A, B, and their product C.

    """
    if matrix_type == "orthogonal_q_qt":
        Q = torch.linalg.qr(torch.randn(n, n, dtype=dtype))[0]
        A = Q
        B = Q.T
        C = torch.eye(n, dtype=dtype)
        return A, B, C
    A, B = generate_pair_matrices(n, l, max_value, dtype, matrix_type, sparsity)
    C = torch.matmul(A, B)
    return A, B, C

def generate_pair_solution_error_matrices(
    n: int,
    l: int,
    max_value: int,
    dtype: torch.dtype,
    matrix_type: str,
    sparsity: float,
    absolute_error: int,
    positive_error: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates a pair of matrices and their product along with an erroneous version of C

    Args:
        n (int): Number of rows for matrix A and columns for matrix B.
        l (int): Number of columns for matrix A and rows for matrix B.
        max_value (int): Maximum value for the matrix elements.
        dtype (torch.dtype): Data type of the matrices.
        matrix_type (str): Type of matrix to generate. Options: 'random', 'sparse_0.1', 'toeplitz', 'diagonal', 'gaussian'.
        sparsity (float): Sparsity level for sparse_0.1 matrices (default: 0.1).
        absolute_error (int): amount of errors in false return
        positive_error (bool): determines if error can only be positiv value

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Generated matrices A, B, and their product C and an erroneous version of C
    """
    A, B = generate_pair_matrices(n, l, max_value, dtype, matrix_type, sparsity)
    C = torch.matmul(A, B)
    C_error = create_matrix_with_errors_torch(C, absolute_error, max_value, positive_error)
    return A, B, C, C_error

def generate_matrix(
    n: int,
    l: int,
    max_value: int,
    dtype: torch.dtype,
    matrix_type: str,
    sparsity: float,
) -> torch.Tensor:
    if matrix_type == 'random':
        return torch.randint(0, 100, (n, l), dtype=dtype)

    elif matrix_type == 'random_max_value_n':
        return torch.randint(0, n, (n, l), dtype=dtype)

    elif matrix_type == 'random_max_value_n^2':
        return torch.randint(0, n * n, (n, l), dtype=dtype)

    elif matrix_type == 'random_signed':
        return torch.randint(-max_value, max_value + 1, (n, l), dtype=dtype)

    elif matrix_type == 'sparse_0.1':
        matrix = torch.zeros((n, l), dtype=dtype)
        num_non_zero = int(n * l * sparsity)
        indices = torch.randperm(n * l)[:num_non_zero]
        values = torch.randint(1, 101, (num_non_zero,), dtype=dtype)  # fixed max = 100
        matrix.view(-1)[indices] = values
        return matrix

    elif matrix_type == 'sparse_0.1_max_value_size':
        matrix = torch.zeros((n, l), dtype=dtype)
        num_non_zero = int(n * l * sparsity)
        indices = torch.randperm(n * l)[:num_non_zero]
        values = torch.randint(1, n + 1, (num_non_zero,), dtype=dtype)
        matrix.view(-1)[indices] = values
        return matrix

    elif matrix_type == 'sparse_0.1_max_value_n^2':
        matrix = torch.zeros((n, l), dtype=dtype)
        num_non_zero = int(n * l * sparsity)
        indices = torch.randperm(n * l)[:num_non_zero]
        values = torch.randint(1, n * n + 1, (num_non_zero,), dtype=dtype)
        matrix.view(-1)[indices] = values
        return matrix

    elif matrix_type == 'sparse_signed':
        matrix = torch.zeros((n, l), dtype=dtype)
        num_non_zero = int(n * l * sparsity)
        indices = torch.randperm(n * l)[:num_non_zero]
        values = torch.randint(-max_value, max_value + 1, (num_non_zero,), dtype=dtype)
        values[values == 0] = 1  # Avoid zero values
        matrix.view(-1)[indices] = values
        return matrix

    elif matrix_type == 'toeplitz':
        c = torch.randint(0, max_value, (n,), dtype=dtype)
        r = torch.randint(0, max_value, (l,), dtype=dtype)
        r[0] = c[0]
        indices = torch.arange(n).view(-1, 1) - torch.arange(l).view(1, -1)
        return torch.where(indices >= 0, c[indices], r[-indices])

    elif matrix_type == 'diagonal':
        diag = torch.randint(0, max_value, (min(n, l),), dtype=dtype)
        return torch.diag(diag)

    elif matrix_type == 'identity':
        return torch.eye(n, l, dtype=dtype)

    elif matrix_type == 'symmetric':
        A = torch.randint(0, max_value, (n, n), dtype=dtype)
        return (A + A.T) // 2

    elif matrix_type == 'triangular':
        A = torch.randint(0, max_value, (n, n), dtype=dtype)
        return torch.triu(A)

    elif matrix_type == 'vandermonde_poly_n^{2}':
        p = mm_verification.calculate_primes(n, 1).item()
        x = torch.randperm(p)[:n]
        V = torch.empty((n, n), dtype=dtype)
        for i in range(n):
            val = 1
            for j in range(n):
                V[i, j] = val
                val = (val * x[i]) % p
        return V

    elif matrix_type == 'ones':
        max_x = int(max_value ** (1 / (l - 1)))
        x = torch.randint(1, max_x + 1, (n,), dtype=dtype)
        return torch.vander(x, l)

    elif matrix_type == 'laplacian':
        A = torch.randint(0, 2, (n, n), dtype=dtype)
        A = (A + A.T) // 2
        D = torch.diag(A.sum(dim=1))
        return D - A

    elif matrix_type == 'hilbert':
        return torch.tensor([[1 / (i + j + 1) for j in range(l)] for i in range(n)], dtype=torch.float32)

    elif matrix_type == 'permutation':
        indices = torch.randperm(n)
        return torch.eye(n, dtype=dtype)[indices]

    elif matrix_type == 'band':
        main_diag = torch.randint(0, max_value, (n,), dtype=dtype)
        off_diag = torch.randint(0, max_value, (n - 1,), dtype=dtype)
        return torch.diag(main_diag) + torch.diag(off_diag, 1) + torch.diag(off_diag, -1)

    elif matrix_type == 'companion':
        coeffs = torch.randint(0, max_value, (n,), dtype=dtype)
        companion = torch.zeros(n, n, dtype=dtype)
        companion[:, -1] = -coeffs[:-1]
        companion[1:, :-1] = torch.eye(n - 1, dtype=dtype)
        return companion

    elif matrix_type == 'nilpotent':
        A = torch.zeros(n, n, dtype=dtype)
        for i in range(n - 1):
            A[i, i + 1] = torch.randint(1, max_value, (1,), dtype=dtype)
        return A

    else:
        raise ValueError(f"Unknown matrix type: {matrix_type}")


def create_matrix_with_errors_torch(
        correct_matrix: torch.Tensor,
        num_errors: int,
        max_value: int,
        only_positive: bool
) -> torch.Tensor:
    """
    Introduces random errors into a given matrix by modifying a specified number of elements.

    Args:
        correct_matrix (torch.Tensor): The original matrix.
        num_errors (int): The number of elements to modify.
        max_value (int): The maximum absolute value for the errors.
        only_positive (bool): If True, errors are only positive.
                              If False, errors are in the range [-max_value, max_value].

    Returns:
        torch.Tensor: The matrix with introduced errors.
    """
    # Clone the original matrix to avoid modifying it
    matrix_with_errors = correct_matrix.clone()

    # Get the shape of the matrix
    rows, cols = matrix_with_errors.shape

    # Ensure the number of errors does not exceed the total number of elements
    num_errors = min(num_errors, rows * cols)

    # Track modified indices to avoid duplicate modifications
    modified_indices = set()

    for _ in range(num_errors):
        # Randomly select a unique index
        while True:
            row_index = random.randint(0, rows - 1)
            col_index = random.randint(0, cols - 1)
            if (row_index, col_index) not in modified_indices:
                modified_indices.add((row_index, col_index))
                break

        # Generate a random error value different from the original value
        original_value = matrix_with_errors[row_index, col_index].item()
        while True:
            if only_positive:
                error_value = random.randint(1, max_value)

            else:
                error_value = random.randint(-max_value, max_value)

            if error_value != original_value:
                break

        # Introduce the error
        matrix_with_errors[row_index, col_index] = error_value

    return matrix_with_errors




def matrix_density(M):
    return (M != 0).sum().item() / M.numel()

def clamp_matrix(M, max_val):
    """Ensure all entries are within [-max_val, max_val]"""
    return torch.clamp(M, -max_val, max_val)

def generate_unimodular_matrix(n, min_density, max_val):
    """Generate a dense, bounded, unimodular matrix using row operations."""
    ops = max(1, int(n * n * min_density))
    while True:
        U = torch.eye(n, dtype=torch.int32)
        for _ in range(ops):
            i, j = random.sample(range(n), 2)
            k = random.randint(-max_val, max_val)
            if k != 0:
                U[i] += k * U[j]
        if matrix_density(U) >= min_density:
            return clamp_matrix(U, max_val)

def generate_diagonal(n, nonzeros, max_val):
    """Generate a sparse_0.1 diagonal matrix with exactly `nonzeros` nonzero entries."""
    diag = [random.randint(1, max_val) if i < nonzeros else 0 for i in range(n)]
    random.shuffle(diag)
    return torch.diag(torch.tensor(diag, dtype=torch.int32))

def generate_triple_for_osmm(n, nonzeros, max_val, min_density=0.8):
    """
    Generate A = U·D, B = V with dense U, V and sparse_0.1 D such that C = AB is sparse_0.1.
    All over ℤ (integers), for output-sensitive matrix multiplication testing.
    """
    assert 0 < nonzeros <= n

    D = generate_diagonal(n, nonzeros, max_val)
    U = generate_unimodular_matrix(n, min_density, max_val)
    V = generate_unimodular_matrix(n, min_density, max_val)

    A = torch.matmul(U, D)         # Dense × sparse_0.1
    B = V                          # Dense
    C = torch.matmul(A, B)         # Final sparse_0.1 product

    return A, B, C



