def fib(n):
    if n < 0:
        raise ValueError("Input must be a non-negative integer")
    if n == 0:
        return 0
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b
