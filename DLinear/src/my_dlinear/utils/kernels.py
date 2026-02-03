def choose_dlinear_kernel(seq_len: int, base: int = 25) -> int:
    """
    Choose an odd kernel size <= seq_len (classic DLinear convenience).
    """
    seq_len = int(seq_len)
    base = int(base)

    k = min(base, seq_len)
    if k <= 1:
        return 1
    if k % 2 == 0:
        k -= 1
    return max(1, k)
