#!/usr/bin/env python3
"""
This module contains a function `minor` to calculate the minor matrix of a
square matrix, and a helper function `determinant` to support that computation.
"""


def determinant(matrix):
    """
    Recursively calculates the determinant of a square matrix.

    Args:
        matrix (list of lists): A square matrix.

    Returns:
        int or float: Determinant of the matrix.
    """
    # Helper function to calculate determinant, reused here
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
    """
    Calculates the minor matrix of a square matrix.

    Args:
        matrix (list of lists): The matrix whose minor matrix is to be 
        calculated.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a non-empty square matrix.

    Returns:
        list of lists: The minor matrix.
    """
    # Check if matrix is a list of lists
    if (not isinstance(matrix, list) or
            not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is non-empty and square
    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)

    # Special case for 1x1 matrix: minor matrix is [[1]]
    if n == 1:
        return [[1]]

    minor_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            # Create submatrix removing row i and column j
            submatrix = [
                matrix[r][:j] + matrix[r][j+1:]
                for r in range(n) if r != i
            ]
            # Calculate determinant of the submatrix
            det_sub = determinant(submatrix)
            row.append(det_sub)
        minor_matrix.append(row)

    return minor_matrix
