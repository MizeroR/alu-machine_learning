def determinant(matrix):
    if matrix == [[]]:
        return 1
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
    det = 0
    for col in range(n):
        minor = [row[:col] + row[col+1:] for row in matrix[1:]]
        det += ((-1) ** col) * matrix[0][col] * determinant(minor)
    return det


def minor(matrix):
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    n = len(matrix)
    if n == 1:
        return [[1]]
    minor_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            submatrix = [matrix[r][:j] + matrix[r][j+1:] for r in range(n) if r != i]
            row.append(determinant(submatrix))
        minor_matrix.append(row)
    return minor_matrix


def cofactor(matrix):
    # Validate input
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Get minor matrix
    minor_matrix = minor(matrix)
    n = len(minor_matrix)

    # Apply sign pattern to minor matrix to get cofactor matrix
    cofactor_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            sign = (-1) ** (i + j)
            row.append(sign * minor_matrix[i][j])
        cofactor_matrix.append(row)

    return cofactor_matrix
