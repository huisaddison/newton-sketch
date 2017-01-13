from optim import *
from predict import *
from helpers import *

import numpy as np

class SketchedNewtonLogisticRegression:
    @auto_assign
    def __init__(self,
            penalty='l2',
            sketch_type='Gaussian',
            sketch_size=None,
            tol=10e-4,
            # fit_intercept=True,
            # class_weight=None,
            max_iter=10,
            lr = 5 * 10e-5,
            ar = None,
            opt = 'GradientDescent',
            gtol = 10e-4,
            gmax_iter = 100,
            verbose=False,
            track_progress=False
            ):
        pass

    def fit(self, A, y):
        if self.sketch_size is None:
            self.sketch_size = int(np.sqrt(A.shape[0]))
        self.coef_, self.nprog_, self.gprog_= run_sketched_newton(
                A, y,
                self.sketch_type, self.sketch_size,
                self.tol, self.max_iter,
                self.ar, self.lr, self.opt, self.gtol, self.gmax_iter,
                self.verbose, self.track_progress)
        return self

    def predict(self, A):
        if not hasattr(self, 'coef_'):
            raise Exception('Call fit before prediction')
        return predict_classes(A, self.coef_)

    def predict_proba(self, A):
        if not hasattr(self, 'coef_'):
            raise Exception('Call fit before prediction')
        return predict_proba(A, self.coef_)

    def score(self, A, y):
        if not hasattr(self, 'coef_'):
            raise Exception('Call fit before prediction')
        preds = predict_classes(A, self.coef_)
        return np.mean(y == preds)

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

# For performance comparison purposes
class GradientDescentLogisticRegression:
    @auto_assign
    def __init__(self,
            penalty='l2',
            sketch_type='Gaussian',
            sketch_size=None,
            tol=10e-4,
            # fit_intercept=True,
            # class_weight=None,
            max_iter=10,
            lr = 5 * 10e-5,
            ar = None,
            opt = 'GradientDescent',
            gtol = 10e-4,
            gmax_iter = 100,
            verbose=False,
            track_progress=False
            ):
        pass

    def fit(self, A, y):
        if self.sketch_size is None:
            self.sketch_size = int(np.sqrt(A.shape[0]))
        self.coef_, self.nprog_, self.gprog_= logis_gradient_descent(
                A, y,
                self.sketch_type, self.sketch_size,
                self.tol, self.max_iter,
                self.ar, self.lr, self.opt, self.gtol, self.gmax_iter,
                self.verbose, self.track_progress)
        return self

    def predict(self, A):
        if not hasattr(self, 'coef_'):
            raise Exception('Call fit before prediction')
        return predict_classes(A, self.coef_)

    def predict_proba(self, A):
        if not hasattr(self, 'coef_'):
            raise Exception('Call fit before prediction')
        return predict_proba(A, self.coef_)

    def score(self, A, y):
        if not hasattr(self, 'coef_'):
            raise Exception('Call fit before prediction')
        preds = predict_classes(A, self.coef_)
        return np.mean(y == preds)
