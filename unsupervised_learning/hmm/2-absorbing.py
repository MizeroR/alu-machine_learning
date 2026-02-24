#!/usr/bin/env python3
"""
Determine if a Markov chain is absorbing
"""

import numpy as np

def absorbing(P):
    """
    Determines if a Markov chain is absorbing.

    P: numpy.ndarray of shape (n, n), transition matrix

    Returns: True if the chain is absorbing, False otherwise
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2 or P.shape[0] != P.shape[1]:
        return False

    n = P.shape[0]

    # A chain is absorbing if it has at least one absorbing state,
    # i.e., P[i,i] == 1, and from every state there is a path to an absorbing state
    absorbing_states = [i for i in range(n) if np.isclose(P[i, i], 1.0)]
    if len(absorbing_states) == 0:
        return False

    # Check if every state can reach an absorbing state
    reachable = np.zeros(n, dtype=bool)
    for a in absorbing_states:
        reachable[a] = True

    changed = True
    while changed:
        changed = False
        for i in range(n):
            if not reachable[i]:
                for j in range(n):
                    if P[i, j] > 0 and reachable[j]:
                        reachable[i] = True
                        changed = True
                        break

    return reachable.all()