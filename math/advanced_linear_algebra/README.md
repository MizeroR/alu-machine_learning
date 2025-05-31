

# Advanced Linear Algebra Utilities

This repository contains Python implementations of various advanced linear algebra operations, including matrix determinant, minor, cofactor, adjugate, inverse, and definiteness checks.

## Directory

`math/advanced_linear_algebra`

---

## Files and Functions

| File                | Function               | Description                                                                                                                                                |
| ------------------- | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `0-determinant.py`  | `determinant(matrix)`  | Computes the determinant of a square matrix. Supports the empty matrix `list [[]]` as 0x0. Raises errors on invalid input.                                 |
| `1-minor.py`        | `minor(matrix)`        | Computes the minor matrix of a non-empty square matrix. Raises errors if input is invalid or not square.                                                   |
| `2-cofactor.py`     | `cofactor(matrix)`     | Computes the cofactor matrix from a non-empty square matrix. Uses minors and applies sign pattern.                                                         |
| `3-adjugate.py`     | `adjugate(matrix)`     | Computes the adjugate (adjoint) matrix, which is the transpose of the cofactor matrix.                                                                     |
| `4-inverse.py`      | `inverse(matrix)`      | Computes the inverse of a non-empty square matrix. Returns `None` if the matrix is singular (determinant = 0).                                             |
| `5-definiteness.py` | `definiteness(matrix)` | Determines the definiteness of a matrix using its eigenvalues (positive definite, semi-definite, negative definite, etc.). Requires `numpy.ndarray` input. |

---

## Usage

Each function validates input types and dimensions before proceeding.

### Example (Determinant)

```python
from 0-determinant import determinant

matrix = [[1, 2], [3, 4]]
print(determinant(matrix))  # Output: -2
```

### Example (Definiteness)

```python
import numpy as np
from 5-definiteness import definiteness

mat = np.array([[5, 1], [1, 1]])
print(definiteness(mat))  # Output: Positive definite
```

---

## Running Tests

Each function has an associated main test file to demonstrate usage:

```bash
./0-main.py
./1-main.py
./2-main.py
./3-main.py
./4-main.py
./5-main.py
```

Make sure to give execute permission to test files:

```bash
chmod +x 0-main.py 1-main.py 2-main.py 3-main.py 4-main.py 5-main.py
```

Run each test script to see the outputs and validate the behavior.

---

## Dependencies

* Python 3.x
* `numpy` (for definiteness checks)

Install numpy via pip if needed:

```bash
pip install numpy
```

---

## Notes

* The empty matrix `list [[]]` is treated as a 0x0 matrix with determinant 1.
* Matrix inputs are expected as lists of lists for all functions except `definiteness`, which requires a numpy ndarray.
* Error handling ensures robust input validation with meaningful messages.

---

## Author

**MizeroR / alu-machine\_learning**