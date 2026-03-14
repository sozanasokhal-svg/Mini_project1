"""
root_finding.py
===============
A numerical root-finding toolbox implemented as a single Python class.

Implemented methods
-------------------
  bisection      – Bisection method
  fixed_point    – Fixed-point iteration
  newton         – Newton's method
  secant         – Secant method
  false_position – Method of False Position (Regula Falsi)
  horner         – Horner's method (polynomial evaluation, used inside solvers)
  muller         – Muller's method (finds complex roots)

Usage
-----
  p = RootFindingProblem(f=my_func, df=my_deriv)
  root = p.solve("newton", x0=1.5)
"""

import cmath


class RootFindingProblem:
    """Numerical root-finding toolbox.

    Parameters
    ----------
    f  : callable, optional
        The function f(x) whose root is sought (f(x) = 0).
    df : callable, optional
        The derivative f'(x); required only for Newton's method.
    g  : callable, optional
        The fixed-point function g(x); required only for fixed-point iteration.
    """

    def __init__(self, f=None, df=None, g=None):
        self.f = f
        self.df = df
        self.g = g

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self, method: str, **kwargs):
        """Dispatch to the chosen root-finding algorithm.

        Parameters
        ----------
        method : str
            One of: 'bisection', 'fixed_point', 'newton', 'secant',
            'false_position', 'horner', 'muller'.
        **kwargs
            Method-specific keyword arguments (see each private method).

        Returns
        -------
        float or complex
            The computed root.
        """
        dispatch = {
            "bisection":      self._bisection,
            "fixed_point":    self._fixed_point,
            "newton":         self._newton,
            "secant":         self._secant,
            "false_position": self._false_position,
            "horner":         self._horner_solver,
            "muller":         self._muller,
        }
        method_lower = method.lower().strip()
        if method_lower not in dispatch:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Choose from: {list(dispatch.keys())}"
            )
        return dispatch[method_lower](**kwargs)

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _bisection(self, a, b, tol=1e-10, max_iter=1000):
        """Bisection method.

        Requires f(a) and f(b) to have opposite signs.

        Parameters
        ----------
        a, b     : float  Bracketing interval [a, b].
        tol      : float  Convergence tolerance (default 1e-10).
        max_iter : int    Maximum number of iterations (default 1000).
        """
        if self.f is None:
            raise ValueError("f(x) must be provided for bisection.")
        fa, fb = self.f(a), self.f(b)
        if fa * fb > 0:
            raise ValueError(
                f"Invalid interval: f(a)={fa} and f(b)={fb} must have opposite signs."
            )
        for i in range(max_iter):
            mid = (a + b) / 2.0
            fmid = self.f(mid)
            if abs(fmid) < tol or (b - a) / 2.0 < tol:
                return mid
            if fa * fmid < 0:
                b = mid
                fb = fmid
            else:
                a = mid
                fa = fmid
        raise RuntimeError(
            f"Bisection did not converge after {max_iter} iterations."
        )

    def _fixed_point(self, x0, tol=1e-10, max_iter=1000):
        """Fixed-point iteration: x_{n+1} = g(x_n).

        Parameters
        ----------
        x0       : float  Initial guess.
        tol      : float  Convergence tolerance (default 1e-10).
        max_iter : int    Maximum number of iterations (default 1000).
        """
        if self.g is None:
            raise ValueError(
                "g(x) must be provided for fixed-point iteration."
            )
        x = x0
        for i in range(max_iter):
            x_new = self.g(x)
            if abs(x_new - x) < tol:
                return x_new
            x = x_new
        raise RuntimeError(
            f"Fixed-point iteration did not converge after {max_iter} iterations."
        )

    def _newton(self, x0, tol=1e-10, max_iter=1000):
        """Newton's (Newton-Raphson) method.

        Parameters
        ----------
        x0       : float  Initial guess.
        tol      : float  Convergence tolerance (default 1e-10).
        max_iter : int    Maximum number of iterations (default 1000).
        """
        if self.f is None:
            raise ValueError("f(x) must be provided for Newton's method.")
        if self.df is None:
            raise ValueError(
                "df(x) (the derivative) must be provided for Newton's method."
            )
        x = x0
        for i in range(max_iter):
            fx = self.f(x)
            dfx = self.df(x)
            if dfx == 0:
                raise ZeroDivisionError(
                    f"Derivative is zero at x={x}; Newton's method fails."
                )
            x_new = x - fx / dfx
            if abs(x_new - x) < tol:
                return x_new
            x = x_new
        raise RuntimeError(
            f"Newton's method did not converge after {max_iter} iterations."
        )

    def _secant(self, x0, x1, tol=1e-10, max_iter=1000):
        """Secant method.

        Parameters
        ----------
        x0, x1   : float  Two initial guesses.
        tol      : float  Convergence tolerance (default 1e-10).
        max_iter : int    Maximum number of iterations (default 1000).
        """
        if self.f is None:
            raise ValueError("f(x) must be provided for the secant method.")
        for i in range(max_iter):
            f0, f1 = self.f(x0), self.f(x1)
            denom = f1 - f0
            if denom == 0:
                raise ZeroDivisionError(
                    f"f(x1) - f(x0) = 0 at iteration {i}; secant method fails."
                )
            x2 = x1 - f1 * (x1 - x0) / denom
            if abs(x2 - x1) < tol:
                return x2
            x0, x1 = x1, x2
        raise RuntimeError(
            f"Secant method did not converge after {max_iter} iterations."
        )

    def _false_position(self, a, b, tol=1e-10, max_iter=1000):
        """Method of False Position (Regula Falsi).

        Parameters
        ----------
        a, b     : float  Bracketing interval [a, b].
        tol      : float  Convergence tolerance (default 1e-10).
        max_iter : int    Maximum number of iterations (default 1000).
        """
        if self.f is None:
            raise ValueError("f(x) must be provided for false position.")
        fa, fb = self.f(a), self.f(b)
        if fa * fb > 0:
            raise ValueError(
                f"Invalid interval: f(a)={fa} and f(b)={fb} must have opposite signs."
            )
        for i in range(max_iter):
            denom = fb - fa
            if denom == 0:
                raise ZeroDivisionError(
                    "Division by zero in false position (fb == fa)."
                )
            c = a - fa * (b - a) / denom
            fc = self.f(c)
            if abs(fc) < tol:
                return c
            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
        raise RuntimeError(
            f"False position did not converge after {max_iter} iterations."
        )

    # ------------------------------------------------------------------
    # Horner's method
    # ------------------------------------------------------------------

    def _horner(self, coeffs, x):
        """Evaluate a polynomial at x using Horner's scheme.

        Parameters
        ----------
        coeffs : list of float
            Coefficients in descending order, i.e.
            coeffs = [a_n, a_{n-1}, ..., a_1, a_0]
            represents  a_n*x^n + ... + a_1*x + a_0.
        x : float or complex
            Point at which to evaluate the polynomial.

        Returns
        -------
        float or complex
            The value of the polynomial at x.
        """
        if not coeffs:
            raise ValueError("Coefficient list must not be empty.")
        result = coeffs[0]
        for coeff in coeffs[1:]:
            result = result * x + coeff
        return result

    def _horner_solver(self, coeffs, a, b, tol=1e-10, max_iter=1000):
        """Find a real root of a polynomial via bisection + Horner evaluation.

        This is the public 'horner' method exposed through solve().

        Parameters
        ----------
        coeffs   : list of float  Polynomial coefficients (descending order).
        a, b     : float          Bracketing interval.
        tol      : float          Convergence tolerance.
        max_iter : int            Maximum iterations.
        """
        fa = self._horner(coeffs, a)
        fb = self._horner(coeffs, b)
        if fa * fb > 0:
            raise ValueError(
                f"Invalid interval: p(a)={fa} and p(b)={fb} must have opposite signs."
            )
        for i in range(max_iter):
            mid = (a + b) / 2.0
            fmid = self._horner(coeffs, mid)
            if abs(fmid) < tol or (b - a) / 2.0 < tol:
                return mid
            if fa * fmid < 0:
                b = mid
                fb = fmid
            else:
                a = mid
                fa = fmid
        raise RuntimeError(
            f"Horner solver did not converge after {max_iter} iterations."
        )

    # ------------------------------------------------------------------
    # Steffensen's method (bonus)
    # ------------------------------------------------------------------

    def _steffensen(self, x0, tol=1e-10, max_iter=1000):
        """Steffensen's method (accelerated fixed-point iteration).

        Parameters
        ----------
        x0       : float  Initial guess.
        tol      : float  Convergence tolerance.
        max_iter : int    Maximum iterations.
        """
        if self.g is None:
            raise ValueError(
                "g(x) must be provided for Steffensen's method."
            )
        x = x0
        for i in range(max_iter):
            gx  = self.g(x)
            ggx = self.g(gx)
            denom = ggx - 2 * gx + x
            if denom == 0:
                raise ZeroDivisionError(
                    f"Steffensen denominator is zero at iteration {i}."
                )
            x_new = x - (gx - x) ** 2 / denom
            if abs(x_new - x) < tol:
                return x_new
            x = x_new
        raise RuntimeError(
            f"Steffensen's method did not converge after {max_iter} iterations."
        )

    # ------------------------------------------------------------------
    # Muller's method
    # ------------------------------------------------------------------

    def _muller(self, x0, x1, x2, tol=1e-10, max_iter=1000):
        """Muller's method – finds real or complex roots.

        Parameters
        ----------
        x0, x1, x2 : float or complex  Three distinct initial guesses.
        tol         : float             Convergence tolerance (default 1e-10).
        max_iter    : int               Maximum iterations (default 1000).

        Returns
        -------
        complex
            The computed root (may be complex).
        """
        if self.f is None:
            raise ValueError("f(x) must be provided for Muller's method.")

        x0, x1, x2 = complex(x0), complex(x1), complex(x2)

        for i in range(max_iter):
            f0, f1, f2 = self.f(x0), self.f(x1), self.f(x2)

            h0 = x1 - x0
            h1 = x2 - x1
            d0 = (f1 - f0) / h0 if h0 != 0 else 0
            d1 = (f2 - f1) / h1 if h1 != 0 else 0

            a = (d1 - d0) / (h1 + h0) if (h1 + h0) != 0 else 0
            b = a * h1 + d1
            c = f2

            discriminant = b * b - 4 * a * c
            sqrt_disc = cmath.sqrt(discriminant)

            denom1 = b + sqrt_disc
            denom2 = b - sqrt_disc
            denom  = denom1 if abs(denom1) >= abs(denom2) else denom2

            if denom == 0:
                raise ZeroDivisionError(
                    f"Muller denominator is zero at iteration {i}."
                )

            dx = -2 * c / denom
            x3 = x2 + dx

            if abs(dx) < tol:
                return x3

            x0, x1, x2 = x1, x2, x3

        raise RuntimeError(
            f"Muller's method did not converge after {max_iter} iterations."
        )
