"""
Computes the approximate fixed point of T via successive approximation.

"""
function successive_approx(T,                     # Operator (callable)
                           x_0;                   # Initial condition
                           tolerance=1e-6,        # Error tolerance
                           max_iter=10_000,       # Max iteration bound
                           print_step=25)         # Print at multiples
    x = x_0
    error = Inf
    k = 1
    while (error > tolerance) & (k <= max_iter)
        x_new = T(x)
        error = maximum(abs.(x_new - x))
        if k % print_step == 0
            println("Completed iteration $k with error $error.")
        end
        x = x_new
        k += 1
    end
    if error <= tolerance
        println("Terminated successfully in $k iterations.")
    else
        println("Termination Warning: Error is greater than tolerance.")
    end
    return x
end
