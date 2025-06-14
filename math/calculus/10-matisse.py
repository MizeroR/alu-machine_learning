#!/usr/bin/env python3
"""
Module to compute the derivative of a polynomial.
"""


def poly_derivative(poly):
    """
    Compute the derivative of a polynomial.

    Args:
        poly (list): List of coefficients, where the index represents
                     the power of x.

    Returns:
        list: Coefficients of the derivative polynomial.
              Returns [0] if the derivative is zero.
              Returns None if input is invalid.
    """
    if (not isinstance(poly, list) or not poly or
            not all(isinstance(x, (int, float)) for x in poly)):
        return None
    if len(poly) <= 1:
        return [0]
    derivative = [coeff * i for i, coeff in enumerate(poly)][1:]
    return derivative if any(derivative) else [0]
