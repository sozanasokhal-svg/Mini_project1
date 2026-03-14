# Root-Finding Toolbox

A pure-Python numerical root-finding library implemented as a single class `RootFindingProblem`.  
No external solvers are used — every algorithm is implemented from scratch using only the Python standard library and `cmath`.

---

## Project Description

This project implements seven classical numerical methods for solving the equation **f(x) = 0**.  
The library exposes a single, clean public interface (`solve(method, **kwargs)`) that dispatches to the appropriate private algorithm.

---

## Implemented Methods

| Method | Key | Notes |
|---|---|---|
| Bisection | `"bisection"` | Bracketing; guaranteed convergence |
| Fixed-Point Iteration | `"fixed_point"` | Requires a contraction mapping g(x) |
| Newton's Method | `"newton"` | Requires f(x) and f'(x) |
| Secant Method | `"secant"` | Derivative-free quasi-Newton |
| False Position (Regula Falsi) | `"false_position"` | Bracketing, never leaves interval |
| Horner's Method | `"horner"` | Polynomial evaluation + bisection root-finder |
| Muller's Method | `"muller"` | Three-point parabolic; finds **complex** roots |

---

## Algorithm Descriptions

### Bisection
Repeatedly halves the interval [a, b] where f(a) and f(b) have opposite signs.  
The midpoint replaces whichever endpoint shares a sign with f(mid).  
Converges at rate **O(1/2ⁿ)**.

### Fixed-Point Iteration
Reformulates f(x)=0 as x = g(x) and iterates x_{n+1} = g(x_n).  
Converges when |g'(x*)| < 1 near the root x*.

### Newton's Method (Newton-Raphson)
Uses the tangent line at x_n to compute the next iterate:  
`x_{n+1} = x_n − f(x_n) / f'(x_n)`  
Converges **quadratically** when started close enough to a simple root.

### Secant Method
Approximates the derivative using a finite difference of the two most recent iterates:  
`x_{n+1} = x_n − f(x_n) · (x_n − x_{n−1}) / (f(x_n) − f(x_{n−1}))`  
Superlinear convergence (~order 1.618); no derivative needed.

### False Position (Regula Falsi)
Like bisection, but replaces the midpoint with the x-intercept of the secant line through (a, f(a)) and (b, f(b)).  
Always stays within the bracket; can stagnate on one side.

### Horner's Method
Evaluates a degree-n polynomial at x in **O(n)** multiplications by nested factoring:  
`p(x) = ((...((aₙx + aₙ₋₁)x + aₙ₋₂)x + ...)x + a₀)`  
Used internally for the `"horner"` solver (bisection with Horner evaluation).

### Muller's Method
Fits a parabola through three points (x₀,f₀), (x₁,f₁), (x₂,f₂) and uses the quadratic formula (choosing the root closer to x₂) as the next iterate.  
Can find **complex roots** and handles polynomials naturally.

---

## File Structure

```
root-finding-project/
├── root_finding.py   # RootFindingProblem class (all algorithms)
├── examples.py       # Runnable demonstrations of every method
└── README.md         # This file
```

---

## How to Run the Examples

No third-party packages required. Python 3.7+ is sufficient.

```bash
# Clone / enter the project folder, then:
python examples.py
```

You should see output like:

```
============================================================
  1. Bisection Method
============================================================
  f(x) = x³ - x - 2  on [1, 2]
  Root ≈ 1.5213797068
  Verify f(root) = 0.00e+00
...
  All examples completed successfully.
```

---

## Quick Code Example

```python
import math
from root_finding import RootFindingProblem

# Define the function and its derivative
f  = lambda x: math.cos(x) - x
df = lambda x: -math.sin(x) - 1

# Create the solver
p = RootFindingProblem(f=f, df=df)

# Newton's method
root = p.solve("newton", x0=0.5)
print(f"Root = {root:.10f}")   # ≈ 0.7390851332

# Bisection (same f, different method)
root2 = p.solve("bisection", a=0, b=1)
print(f"Root = {root2:.10f}")  # ≈ 0.7390851332

# Secant (no derivative needed)
p2 = RootFindingProblem(f=f)
root3 = p2.solve("secant", x0=0, x1=1)
print(f"Root = {root3:.10f}")  # ≈ 0.7390851332

# Polynomial root via Horner (coefficients in descending order)
p3 = RootFindingProblem()
# p(x) = x³ - x - 2  →  coeffs = [1, 0, -1, -2]
root4 = p3.solve("horner", coeffs=[1, 0, -1, -2], a=1, b=2)
print(f"Root = {root4:.10f}")  # ≈ 1.5213797068

# Complex root via Muller
f_complex = lambda x: x**2 + 1   # roots are ±i
p4 = RootFindingProblem(f=f_complex)
root5 = p4.solve("muller", x0=0, x1=1, x2=2)
print(f"Root = {root5}")          # ≈ 0+1j
```

---

## Error Handling

| Situation | Exception raised |
|---|---|
| Invalid interval (same-sign endpoints) | `ValueError` |
| Missing derivative for Newton | `ValueError` |
| Missing g(x) for fixed-point | `ValueError` |
| Division by zero during iteration | `ZeroDivisionError` |
| No convergence within max_iter | `RuntimeError` |
| Unknown method string | `ValueError` |
