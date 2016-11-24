import numpy as np

def gen_sketch_mat(m, n, method):
    """Generate a sketch matrix.

    A sketch matrix $S\in\mathbb{R}^{m\times n}$ has the property that
    $\mathbb{E} S^T S = \mathbb{I}/m$.  

    Args:
        m (int): number of rows of the sketch matrix (desired rank)
        n (int): number of columns of the sketch matrix (size of matrix
            to be sketched)
        method (str): method for generating the sketch matrix.  Currently,
            only random normal sketch matrices are supported.
    Returns:
        np.ndarray: a sketch matrix
    """
    if method is 'Gaussian':
        S = np.random.randn(m, n) / m
    elif method is 'Rademacher':
        # Produces r.v. in {0, 1} with equal probability
        S = ((np.random.randn(m, n) > 0) * 2 - 1) / m
    else:
        raise ValueError('Unrecognized sketch type: ' + method)
    return S

