#!/usr/bin/env python3
"""
Calculates the integral of a polynomial represented as a list of coefficients.
"""

def poly_integral(poly, C=0):
    """Compute the integral of the polynomial represented by poly.

    Args:
        poly (list): List of coefficients where index is power of x.
        C (int): Integration constant.

    Returns:
        list or None: Integral polynomial coefficients list, or None if input invalid.
    """
    # Validate inputs
    if (not isinstance(poly, list) or
            not all(isinstance(x, (int, float)) for x in poly)):
        return None
    if not isinstance(C, int):
        return None

    integral = [C]  # start with integration constant at index 0

    for i, coef in enumerate(poly):
        new_coef = coef / (i + 1)
        # Convert to int if whole number
        if new_coef.is_integer():
            new_coef = int(new_coef)
        integral.append(new_coef)

    # Remove trailing zeros to keep list as small as possible
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
