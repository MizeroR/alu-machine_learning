#!/usr/bin/env python3
def mat_mul(mat1, mat2):
    # Check if matrices can be multiplied: columns in mat1 == rows in mat2
    if len(mat1[0]) != len(mat2):
        return None

    # Initialize result matrix with zeros
    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            # Calculate the dot product of the i-th row of mat1 and j-th column of mat2
            s = 0
            for k in range(len(mat2)):
                s += mat1[i][k] * mat2[k][j]
            row.append(s)
        result.append(row)

    return result