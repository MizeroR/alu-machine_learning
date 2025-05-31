import numpy as np

def definiteness(matrix):
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    
    # Check if matrix is square and 2D
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    # Check if matrix is empty or invalid shape
    if matrix.size == 0:
        return None

    # Compute eigenvalues
    try:
        eigvals = np.linalg.eigvalsh(matrix)  # For symmetric/hermitian matrices, more stable
    except np.linalg.LinAlgError:
        return None

    pos = np.all(eigvals > 0)
    pos_semi = np.all(eigvals >= 0)
    neg = np.all(eigvals < 0)
    neg_semi = np.all(eigvals <= 0)

    if pos:
        return "Positive definite"
    if pos_semi and not pos:
        return "Positive semi-definite"
    if neg:
        return "Negative definite"
    if neg_semi and not neg:
        return "Negative semi-definite"

    # If none of the above
    return "Indefinite"
