include("s_approx.jl")
using LinearAlgebra

# Compute the fixed point of Tx = Ax + b via linear algebra
A, b = [0.4 0.1; 0.7 0.2], [1.0; 2.0]
x_star = (I - A) \ b  # compute (I - A)^{-1} * b

# Compute the fixed point via successive approximation
T(x) = A * x + b
x_0 = [1.0; 1.0]
x_star_approx = successive_approx(T, x_0)

# Test for approximate equality (prints "true")
print(isapprox(x_star, x_star_approx, rtol=1e-5))

