#!/usr/bin/env python3
"""
Module that provides a function to calculate the
summation of squares from 1 to n using a formula.
"""


def summation_i_squared(n):
    """
    Calculate the summation of squares from 1 to n.

    Args:
        n (int): The upper bound of summation.

    Returns:
        int: Sum of squares from 1 to n.
             Returns None if n is not a valid positive integer.
    """
    if not isinstance(n, int) or n < 1:
        return None
    return n * (n + 1) * (2 * n + 1) // 6
