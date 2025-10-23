def fib(n: int) -> int:
    """
    Calculates the nth Fibonacci number efficiently using a fast doubling algorithm.
    This approach has a time complexity of O(log n).
    """
    if not isinstance(n, int) or n < 0:
        raise ValueError("Input must be a non-negative integer.")

    if n <= 1:
        return n

    # The algorithm uses the identities:
    # F(2k) = F(k) * [2*F(k+1) - F(k)]
    # F(2k+1) = F(k+1)^2 + F(k)^2
    
    a, b = 0, 1  # Represents F(k), F(k+1) starting with k=0
    
    # Iterate from the most significant bit of n down to the least.
    for i in range(n.bit_length() - 1, -1, -1):
        # Double the current F(k), F(k+1) to get F(2k), F(2k+1)
        c = a * (2 * b - a)  # F(2k)
        d = a**2 + b**2      # F(2k+1)

        # If the current bit of n is 1, we advance one step further
        if (n >> i) & 1:
            # F(2k+1), F(2k+2) = d, c + d
            a, b = d, c + d
        else:
            # F(2k), F(2k+1) = c, d
            a, b = c, d
            
    return a
