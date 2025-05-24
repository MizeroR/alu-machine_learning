#!/usr/bin/env python3
"""
This module provides a function to add two vectors element-wise.
"""


def add_arrays(arr1, arr2):
    """
    Adds two arrays element-wise.

    Args:
        arr1 (list): The first list of integers/floats.
        arr2 (list): The second list of integers/floats.

    Returns:
        list: A new list containing the element-wise sums,
              or None if the arrays are not the same length.
    """
    if len(arr1) != len(arr2):
        return None
    return [a + b for a, b in zip(arr1, arr2)]
