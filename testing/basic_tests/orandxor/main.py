import sys
sys.setrecursionlimit(1000000)

def lcg(x: int) -> int:
    # constants chosen to stay in 32-bit-ish range
    return (1103515245 * x + 12345) % 2147483647

def gcd(a: int, b: int) -> int:
    if b == 0:
        return a
    else:
        return gcd(b, a % b)

def bench(n: int, x: int, acc: int) -> int:
    if n == 0:
        return acc
    else:
        x1 = lcg(x)
        a  = x1
        x2 = lcg(x1)
        b  = x2
        g  = gcd(a, b)
        return bench(n - 1, x2, acc + g)

print(bench(20000, 123456789, 0))
