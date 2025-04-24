import random
import torch
import os
from scipy.io import mmread
from typing import Tuple

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

    # Step 2: Convert the sparse matrix to a dense NumPy array
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
    dtype,
    sparsity,
    nonzero_fn_map=None  # Optional: dict from matrix_type to custom nonzero_fn
):
    """
    Extended version to support 'rank1_sqrt(n)_nonzeroes' type which ensures t nonzeros in AB.

    Args:
        matrix_types (List[str]): List of matrix types.
        sizes (List[int]): List of matrix sizes.
        max_value (int): Max value of matrix entries.
        dtype (torch.dtype): PyTorch dtype.
        sparsity (float): For standard sparse generation or nonzero scaling.
        nonzero_fn_map (Dict[str, Callable[[int], int]]): Custom nonzero functions per matrix_type.
    """
    if not os.path.exists("data_test_int"):
        os.makedirs("data_test_int")

    for matrix_type in matrix_types:
        print(f"Generating {matrix_type} matrices...")
        for size in sizes:
            for u in range(2):  # Two instances per size
                instance_dir = os.path.join("data_test_int", matrix_type, f"u_{u}")
                os.makedirs(instance_dir, exist_ok=True)

                if matrix_type == "rank1_sqrt(n)_nonzeroes":
                    nonzero_fn = (
                        nonzero_fn_map.get(matrix_type)
                        if nonzero_fn_map and matrix_type in nonzero_fn_map
                        else lambda n: int(sparsity * n)  # default
                    )
                    A, B, C = generate_rank1_product_controlled(
                        n=size,
                        max_value=max_value,
                        dtype=dtype,
                        nonzero_fn=nonzero_fn
                    )
                    save_matrices(
                        A, B, C,
                        save_dir=instance_dir,
                        name_A=f"A_size_{size}.pt",
                        name_B=f"B_size_{size}.pt",
                        name_C=f"C_size_{size}.pt"
                    )
                else:
                    generate_and_save_matrices(
                        size, size, max_value, dtype,
                        matrix_type, sparsity,
                        instance_dir,
                        f"A_size_{size}.pt",
                        f"B_size_{size}.pt",
                        f"C_size_{size}.pt"
                    )

def create_file_structure_for_test_data(
        matrix_types,
        sizes,
        max_value,
        dtype,
        sparsity
):
    """
    Method which creates file structure and generates and loads matrices for a list of types nad sizes
    :param matrix_types: kind of matrices which get generated
    :param sizes: sizes of matrices which get created
    :param max_value: max_value
    :param dtype: data_type
    :param sparsity: sparsity
    """
    if not os.path.exists("data_test_int"):
        os.makedirs("data_test_int")

    for matrix_type in matrix_types:
        print(f"Generating {matrix_type} matrices...")
        for size in sizes:
            for u in range(2):  # Generate 2 instances of each size
                # Create subdirectory for each matrix type and instance
                instance_dir = os.path.join("data_test_int", matrix_type, f"u_{u}")
                os.makedirs(instance_dir, exist_ok=True)

                # Generate and save matrices
                generate_and_save_matrices(
                    size,
                    size,
                    max_value,
                    dtype,
                    matrix_type,
                    sparsity,
                    instance_dir,
                    f"A_size_{size}.pt",
                    f"B_size_{size}.pt",
                    f"C_size_{size}.pt"
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
    matrix_type: str = 'random',
    sparsity: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a pair of matrices of the specified type.

    Args:
        n (int): Number of rows for matrix A and columns for matrix B.
        l (int): Number of columns for matrix A and rows for matrix B.
        max_value (int): Maximum value for the matrix elements.
        dtype (torch.dtype): Data type of the matrices.
        matrix_type (str): Type of matrix to generate. Options: 'random', 'sparse', 'toeplitz', 'diagonal', 'gaussian'.
        sparsity (float): Sparsity level for sparse matrices (default: 0.1).

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
        matrix_type (str): Type of matrix to generate. Options: 'random', 'sparse', 'toeplitz', 'diagonal', 'gaussian'.
        sparsity (float): Sparsity level for sparse matrices (default: 0.1).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Generated matrices A, B, and their product C.
    """
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
        matrix_type (str): Type of matrix to generate. Options: 'random', 'sparse', 'toeplitz', 'diagonal', 'gaussian'.
        sparsity (float): Sparsity level for sparse matrices (default: 0.1).
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
    """
    Generates a matrix of a specified type.

    Args:
        n (int): Number of rows.
        l (int): Number of columns.
        max_value (int): Maximum value for the matrix elements.
        dtype (torch.dtype): Data type of the matrix.
        matrix_type (str): Type of matrix to generate.
        sparsity (float): Sparsity level for sparse matrices (default: 0.1).

    Returns:
        torch.Tensor: Generated matrix.
    """
    # Random Matrix: A matrix with random integer values between 0 and max_value.
    if matrix_type == 'random':
        return torch.randint(0, max_value, (n, l), dtype=dtype)

    # Sparse Matrix: A matrix with a specified sparsity level (most elements are zero).
    elif matrix_type == 'sparse':
        matrix = torch.zeros((n, l), dtype=dtype)
        num_non_zero = int(n * l * sparsity)
        indices = torch.randperm(n * l)[:num_non_zero]
        values = torch.randint(1, max_value, (num_non_zero,), dtype=dtype)
        matrix.view(-1)[indices] = values
        return matrix

    # Toeplitz Matrix: A matrix where each descending diagonal is constant.
    elif matrix_type == 'toeplitz':
        c = torch.randint(0, max_value, (n,), dtype=dtype)  # First column
        r = torch.randint(0, max_value, (l,), dtype=dtype)  # First row

        # Ensure the first element of r is the same as c (Toeplitz property)
        r[0] = c[0]

        # Create Toeplitz matrix using broadcasting
        indices = torch.arange(n).view(-1, 1) - torch.arange(l).view(1, -1)
        toeplitz_matrix = torch.where(indices >= 0, c[indices], r[-indices])

        return toeplitz_matrix

    # Diagonal Matrix: A matrix with non-zero elements only on the main diagonal.
    elif matrix_type == 'diagonal':
        diagonal_values = torch.randint(0, max_value, (min(n, l),), dtype=dtype)
        return torch.diag(diagonal_values)

    # Gaussian Matrix: A matrix with elements drawn from a Gaussian (normal) distribution.
    elif matrix_type == 'gaussian':
        return torch.round(torch.randn(n, l) * max_value).to(torch.int)
    # Identity Matrix: A square matrix with 1s on the diagonal and 0s elsewhere.
    elif matrix_type == 'identity':
        return torch.eye(n, l, dtype=dtype)

    # Symmetric Matrix: A square matrix that is equal to its transpose (A = A^T).
    elif matrix_type == 'symmetric':
        A = torch.randint(0, max_value, (n, n), dtype=dtype)
        return (A + A.T) // 2  # Ensure symmetry

    # Triangular Matrix: A matrix with zeros below (upper) or above (lower) the diagonal.
    elif matrix_type == 'triangular':
        A = torch.randint(0, max_value, (n, n), dtype=dtype)
        return torch.triu(A)  # Upper triangular

    # Orthogonal Matrix: A square matrix with orthonormal columns and rows (Q^T * Q = I).
    elif matrix_type == 'orthogonal':
        Q, _ = torch.qr(torch.rand(n, n, dtype=dtype))
        return Q

    # Hankel Matrix: A matrix where each ascending skew-diagonal is constant.
    elif matrix_type == 'hankel':
        c = torch.randint(0, max_value, (n,), dtype=dtype)
        r = torch.randint(0, max_value, (l,), dtype=dtype)
        return torch.hankel(c, r)

    # Vandermonde Matrix: A matrix where each row is a geometric progression.
    elif matrix_type == 'vandermonde':
        max_x = int(max_value ** (1 / (l - 1)))  # Compute the maximum allowed value for x
        x = torch.randint(1, max_x + 1, (n,), dtype=dtype)  # Ensure x_i <= max_x
        return torch.vander(x, l)

    # Laplacian Matrix: A matrix representing the graph Laplacian (L = D - A).
    elif matrix_type == 'laplacian':
        A = torch.randint(0, 2, (n, n), dtype=dtype)
        A = (A + A.T) // 2  # Make it symmetric (undirected graph)
        D = torch.diag(A.sum(dim=1))
        return D - A

    # Hilbert Matrix: A matrix with entries H[i][j] = 1 / (i + j + 1).
    elif matrix_type == 'hilbert':
        return torch.tensor([[1 / (i + j + 1) for j in range(l)] for i in range(n)], dtype=dtype)

    # Permutation Matrix: A matrix obtained by permuting the rows of an identity matrix.
    elif matrix_type == 'permutation':
        indices = torch.randperm(n)
        return torch.eye(n, dtype=dtype)[indices]

    # Positive Definite Matrix: A symmetric matrix with all positive eigenvalues.
    elif matrix_type == 'positive_definite':
        A = torch.randn(n, n, dtype=dtype)
        return A @ A.T  # Ensure positive definiteness

    # Stochastic Matrix: A matrix where each row sums to 1 (used in probability).
    elif matrix_type == 'stochastic':
        A = torch.rand(n, l, dtype=dtype)
        return A / A.sum(dim=1, keepdim=True)  # Row normalization

    # Band Matrix: A sparse matrix with non-zero elements confined to a diagonal band.
    elif matrix_type == 'band':
        main_diag = torch.randint(0, max_value, (n,), dtype=dtype)
        off_diag = torch.randint(0, max_value, (n - 1,), dtype=dtype)
        return torch.diag(main_diag) + torch.diag(off_diag, diagonal=1) + torch.diag(off_diag, diagonal=-1)

    # Cauchy Matrix: A matrix with entries C[i][j] = 1 / (x_i - y_j).
    elif matrix_type == 'cauchy':
        x = torch.randint(1, max_value, (n,), dtype=dtype)
        y = torch.randint(1, max_value, (l,), dtype=dtype)
        return torch.tensor([[1 / (x[i] - y[j]) for j in range(l)] for i in range(n)], dtype=dtype)

    # Companion Matrix: A matrix associated with a polynomial (used to find its roots).
    elif matrix_type == 'companion':
        coeffs = torch.randint(0, max_value, (n,), dtype=dtype)
        companion = torch.zeros(n, n, dtype=dtype)
        companion[:, -1] = -coeffs[:-1]
        companion[1:, :-1] = torch.eye(n - 1, dtype=dtype)
        return companion

    # Nilpotent Matrix: A matrix where some power of the matrix is the zero matrix.
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
                error_value = random.randint(0, max_value)
            else:
                error_value = random.randint(-max_value, max_value)

            if error_value != original_value:
                break

        # Introduce the error
        matrix_with_errors[row_index, col_index] = error_value

    return matrix_with_errors






