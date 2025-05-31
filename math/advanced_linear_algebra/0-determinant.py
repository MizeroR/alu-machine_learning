#!/usr/bin/env python3
"""
This module contains a function `determinant` to calculate the determinant
of a square matrix represented as a list of lists.
"""


def determinant(matrix):
    """
    Calculates the determinant of a square matrix.

    Args:
        matrix (list of lists): The square matrix to calculate the 
        determinant of.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not square.

    Returns:
        int or float: The determinant of the matrix.
    """

    # Check if matrix is a list of lists
    if (not isinstance(matrix, list) or
            not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    # Handle 0x0 matrix case: matrix = [[]]
    if matrix == [[]]:
        return 1

    # Check for empty matrix or inconsistent row sizes
    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")

    n = len(matrix)

    # Base case for 1x1 matrix
    if n == 1:
        return matrix[0][0]

    # Base case for 2x2 matrix
    if n == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    # Recursive case for nxn matrix
    det = 0
    for col in range(n):
        # Create minor matrix by excluding first row and current column
        minor = [row[:col] + row[col+1:] for row in matrix[1:]]
        # Calculate cofactor and recurse
        det += ((-1) ** col) * matrix[0][col] * determinant(minor)

    return det
