import numpy as np

def univariate_gaussian (mean, std):
    u1 = np.random.uniform(0, 1)
    u2 = np.random.uniform(0, 1)
    z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    
    return mean + np.sqrt(std) * z1

def polynomial_basis(n, a, w):
    assert len(w) == n, "Length of w must be equal to n"

    x = np.random.uniform(-1, 1)
    phi = np.stack([x**i for i in range(n)])
    e = univariate_gaussian(0, a)
    
    return x, float(np.dot(w, phi) + e)


if __name__ == "__main__":
    m = 0
    s = 1
    print(univariate_gaussian(m, s))

    n = 3
    a = 1
    w = [1, 2, 3]
    print(polynomial_basis(n, a, w))
