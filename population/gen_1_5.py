def fib(n):
    """
    Calculates the nth Fibonacci number using an efficient iterative approach.
    Handles non-negative integers as input.
    """
    if n < 0:
        raise ValueError("Input must be a non-negative integer.")
    
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
