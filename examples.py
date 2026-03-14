"""
examples.py
===========
Demonstrations of every method in RootFindingProblem.

Run with:
    python examples.py
"""

import cmath
from root_finding import RootFindingProblem

# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Bisection  –  f(x) = x³ - x - 2,  root ≈ 1.52138
# ──────────────────────────────────────────────────────────────────────────────

section("1. Bisection Method")

f1  = lambda x: x**3 - x - 2
p1  = RootFindingProblem(f=f1)
r1  = p1.solve("bisection", a=1, b=2)
print(f"  f(x) = x³ - x - 2  on [1, 2]")
print(f"  Root ≈ {r1:.10f}")
print(f"  Verify f(root) = {f1(r1):.2e}")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Fixed-Point Iteration  –  g(x) = (x + 2)^(1/3),  root of x³ - x - 2 = 0
# ──────────────────────────────────────────────────────────────────────────────

section("2. Fixed-Point Iteration")

g2  = lambda x: (x + 2) ** (1/3)
p2  = RootFindingProblem(g=g2)
r2  = p2.solve("fixed_point", x0=1.5)
print(f"  g(x) = (x+2)^(1/3),  x0=1.5")
print(f"  Root ≈ {r2:.10f}")
print(f"  Verify x - g(x) = {r2 - g2(r2):.2e}")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Newton's Method  –  f(x) = cos(x) - x,  root ≈ 0.73909
# ──────────────────────────────────────────────────────────────────────────────

section("3. Newton's Method")

import math
f3  = lambda x: math.cos(x) - x
df3 = lambda x: -math.sin(x) - 1
p3  = RootFindingProblem(f=f3, df=df3)
r3  = p3.solve("newton", x0=0.5)
print(f"  f(x) = cos(x) - x,  x0=0.5")
print(f"  Root ≈ {r3:.10f}")
print(f"  Verify f(root) = {f3(r3):.2e}")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Secant Method  –  f(x) = e^x - 3x,  root ≈ 1.51213
# ──────────────────────────────────────────────────────────────────────────────

section("4. Secant Method")

f4  = lambda x: math.exp(x) - 3*x
p4  = RootFindingProblem(f=f4)
r4  = p4.solve("secant", x0=1, x1=2)
print(f"  f(x) = e^x - 3x,  x0=1, x1=2")
print(f"  Root ≈ {r4:.10f}")
print(f"  Verify f(root) = {f4(r4):.2e}")

# ──────────────────────────────────────────────────────────────────────────────
# 5. False Position  –  f(x) = x³ - x - 2  on [1, 2]
# ──────────────────────────────────────────────────────────────────────────────

section("5. Method of False Position (Regula Falsi)")

f5  = lambda x: x**3 - x - 2
p5  = RootFindingProblem(f=f5)
r5  = p5.solve("false_position", a=1, b=2)
print(f"  f(x) = x³ - x - 2  on [1, 2]")
print(f"  Root ≈ {r5:.10f}")
print(f"  Verify f(root) = {f5(r5):.2e}")

# ──────────────────────────────────────────────────────────────────────────────
# 6. Horner's Method  –  p(x) = 2x⁴ - 3x³ + x² - 5x + 7, root in [-2, -1]
# ──────────────────────────────────────────────────────────────────────────────

section("6. Horner's Method (polynomial root via bisection + Horner evaluation)")

# coefficients in descending order: [2, -3, 1, -5, 7]
# p(x) = 2x⁴ - 3x³ + x² - 5x + 7
coeffs = [2, -3, 1, -5, 7]

# Verify sign change to ensure bracket is valid
p6   = RootFindingProblem()
pa   = p6._horner(coeffs, -2)
pb   = p6._horner(coeffs, -1)
print(f"  p(x) = 2x⁴ - 3x³ + x² - 5x + 7  on [-2, -1]")
print(f"  p(-2) = {pa:.4f},  p(-1) = {pb:.4f}")

if pa * pb < 0:
    r6 = p6.solve("horner", coeffs=coeffs, a=-2, b=-1)
    print(f"  Root ≈ {r6:.10f}")
    print(f"  Verify p(root) = {p6._horner(coeffs, r6):.2e}")
else:
    # Demonstrate Horner evaluation even if no sign change in that bracket
    print("  (No sign change in [-2,-1]; demonstrating Horner evaluation instead)")
    val = p6._horner(coeffs, 1.5)
    print(f"  p(1.5) via Horner = {val:.6f}")
    val_direct = 2*(1.5)**4 - 3*(1.5)**3 + (1.5)**2 - 5*(1.5) + 7
    print(f"  p(1.5) direct     = {val_direct:.6f}  (should match)")

# ──────────────────────────────────────────────────────────────────────────────
# 7. Muller's Method  –  p(x) = x³ - 1  (complex cube roots of unity)
# ──────────────────────────────────────────────────────────────────────────────

section("7. Muller's Method (complex root)")

# x³ - 1 = 0 has three roots: 1, and the two complex ones
# e^(±2πi/3) = -0.5 ± 0.866i
f7  = lambda x: x**3 - 1
p7  = RootFindingProblem(f=f7)

# Starting near the complex root
r7  = p7.solve("muller", x0=0+1j, x1=-0.5+0.9j, x2=-1+0.5j)
print(f"  f(x) = x³ - 1")
print(f"  Root ≈ {r7.real:.8f} + {r7.imag:.8f}i")
print(f"  Verify |f(root)| = {abs(f7(r7)):.2e}")
expected = complex(-0.5, math.sqrt(3)/2)
print(f"  Expected          {expected.real:.8f} + {expected.imag:.8f}i")

# ──────────────────────────────────────────────────────────────────────────────
# Done
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  All examples completed successfully.")
print("="*60 + "\n")
