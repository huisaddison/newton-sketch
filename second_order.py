import numpy as np

def logis_laplacian_weights(A, x):
    """Returns the weights of the Hessian of the (negative)
    log-likelihood.

    The Hessian of the negative log-likelihood may be rewritten as:
    $$
        A^T W A
    $$
    where $W$ is a diagonal matrix, with:
    $$
        W_{ii} = p(a_i; x)(1 - p(a_i; x) = \frac{1}{(1 + \exp(a_i^T x))^2}
    $$

    Args:
        A (np.ndarray): Feature matrix
        x (np.ndarray): Coefficients

    Returns:
        np.ndarray: the diagonal of the weight matrix $W$ as defined above.
    """
    e = np.exp(A.dot(x))
    return e / ((1 + e) ** 2)

def logis_sketched_hessian_sqrt(A, x, y, S):
    """Returns the square root of the sketched Hessian for the negative
    log-likelihood.
    
    The sketched Hessian takes the form
    $$
        A^T S^T W S A
    $$
    and because of its diagonal form, it may be represented as:
    $$
        B^T B
    $$
    where
    $$
        B = SAW^{\frac{1}{2}}
    $$

    Args:
        A (np.ndarray): Feature matrix
        x (np.ndarray): Coefficients
        y (np.ndarray): Targets
        S (np.ndarray): Sketch matrix
        
    Returns:
        np.ndarray: square root of Hessian, as described above as $B$.
    """
    # Reshape into column vector to broadcast across rows
    d = (logis_laplacian_weights(A, x,) ** 0.5).reshape((-1, 1))
    return S.dot(A * d)


