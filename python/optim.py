from functools import partial
from time import time

from loss_funcs import *
from second_order import *
from gen_sketch import *

import numpy as np

def line_search(x, dx, g, dg, a, b):
    """Perform backtracking line search.

    Backtracking line search begins with an initial step-size dx and backtracks
    until the adjusted linear estimate overestimates the loss function $g$.
    For more information refer to pgs. 464-466 of Convex Optimization by Boyd.

    Args:
        x (np.ndarray): Coefficients
        dx (np.ndarray): Step direction
        g (function): Loss function
        dg (function): Loss function gradient
        a (numeric): scaling factor
        b (numeric): reduction factor

    Returns:
        float
    """
    mu = 1
    while (g(x=x) + mu * a * dg(x=x).T.dot(dx)
            < g(x=x + mu * dx)):
        mu = mu * b
    return mu

def newton_sketch_step(A, x, dx, y, m, n):
    a, b = 0.1, 0.5
    g = partial(logis_loss, A=A, y=y)
    dg = partial(logis_loss_grad, A=A, y=y)
    mu = line_search(x, dx, g, dg, a, b)
    return x + mu * dx

def run_sketched_newton(
        A, y,
        sketch_type, m,
        ntol, nmax_iter,
        ar, lr, opt, gtol, gmax_iter,
        verbose, track_progress):
    n = A.shape[0]
    p = A.shape[1]

    # Initialize weights vector
    x = np.zeros(p).reshape(-1, 1)

    if track_progress:
        nprog = np.zeros([nmax_iter, 4])
        gprog = np.zeros([gmax_iter, nmax_iter, 2])
    else:
        nprog = None
        gprog = None

    if verbose:
        print('t               loss  gradient')
        print('--         ---------  --------')
    for t in range(nmax_iter):
        # Track total time per iteration (without gradient step)
        ntime = 0
        nstart = time()
        dx = np.zeros(p).reshape(-1, 1)
        S = gen_sketch_mat(m, n, sketch_type)
        # Create sketched Hessian square root
        if sketch_type is False:
            d = (logis_laplacian_weights(A, x,) ** 0.5).reshape((-1, 1))
            B = A * d
        else:
            B = logis_sketched_hessian_sqrt(A, x, y, S)
        # Get gradient of logistic loss
        grad = logis_loss_grad(A, x, y)
        # Add time elapsed
        ntime += time() - nstart

        gtime = 0
        if opt is 'AdaGrad':
            grad_history = np.zeros((p, 1))
        for i in range(gmax_iter):
            if track_progress:
                # Time at start of loop
                gstart = time()
            # Calculate Gradient
            nl_grad = newton_loss_grad(dx, grad, B)

            # Decide Optimizer
            if opt is 'GradientDescent':
                lr_ = lr / np.sqrt(i+1)
                lr_ = lr * np.sqrt(t+1) / np.sqrt(i+1)
            elif opt is 'AdaGrad':
                grad_history += nl_grad ** 2
                lr_ = lr / (np.sqrt(grad_history) + 1e-8)
            else:
                raise ValueError('Unrecognized optimizer: ' + opt)

            # Update
            dx = dx - lr_ * nl_grad

            grad_norm = np.linalg.norm(nl_grad) / p
            if track_progress:
                gtime += time() - gstart
                gprog[i, t, 0] = newton_loss(dx, grad, B) / n
                gprog[i, t, 1] = grad_norm
            if grad_norm < 1e-2:
                break

        # Time this block too
        nstart = time()
        x = newton_sketch_step(A, x, dx, y, m, n)
        loss = logis_loss(A, x, y) / n
        gsum = np.linalg.norm(logis_loss_grad(A, x, y)) / p
        # Time this block too
        ntime += time() - nstart
        if verbose:
            print('{:2d}         {:9.6f}  {}'.format(t, loss, gsum))
        if track_progress:
            nprog[t, :] = loss, gsum, ntime, gtime
    return x, nprog, gprog

def logis_gradient_descent(
        A, y,
        sketch_type, m,
        ntol, nmax_iter,
        ar, lr, opt, gtol, gmax_iter,
        verbose, track_progress):
    n = A.shape[0]
    p = A.shape[1]

    x = np.zeros(p).reshape(-1, 1)

    if track_progress:
        nprog = np.zeros([nmax_iter, 3])
        gprog = np.zeros([gmax_iter, nmax_iter, 2])
    else:
        nprog = None
        gprog = None

    if verbose:
        print('t               loss  gradient')
        print('--         ---------  --------')
    for t in range(nmax_iter):
        t0 = time()
        dx = np.zeros(p).reshape(-1, 1)
        grad = logis_loss_grad(A, x, y)

        x = x - (lr / np.sqrt(t+1)) * grad
        loss = logis_loss(A, x, y) / n
        gsum = np.linalg.norm(logis_loss_grad(A, x, y)) / p
        
        ntime = time() - t0
        if verbose:
            print('{:2d}         {:9.6f}  {}'.format(t, loss, gsum))
        if track_progress:
            nprog[t, :] = loss, gsum, ntime
    return x, nprog, gprog

