#!/usr/bin/env python3
def cat_matrices2D(mat1, mat2, axis=0):
    if axis == 0:
        # Check if columns match
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2
    elif axis == 1:
        # Check if rows match
        if len(mat1) != len(mat2):
            return None
        return [r1 + r2 for r1, r2 in zip(mat1, mat2)]
    else:
        return None