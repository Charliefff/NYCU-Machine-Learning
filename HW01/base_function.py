def polynomial_basis(x: list, degree: int) -> list:
    basis = [[xi ** i for i in range(degree)] for xi in x]  
    return basis  # (len(x), degree)

def build_zero_matrix(n: int, m: int) -> list:
    return [[0] * m for _ in range(n)]

def transpose(A: list) -> list:
    assert check_2D_list(A), "Input must be a 2D matrix."
    
    rows, cols = len(A), len(A[0])  
    transposed = build_zero_matrix(cols, rows) 
    
    for i in range(rows):
        for j in range(cols):
            transposed[j][i] = A[i][j] 
    
    return transposed

def matrix_multiplication(A: list, B: list) -> list:
    # A or B is 1D
    if isinstance(A[0], (int, float)):
        A = [[a] for a in A]
    if isinstance(B[0], (int, float)):
        B = [[b] for b in B]
        
    assert len(A[0]) == len(B), "A row must == B col"
    # if len(B) 
    rowA, colA = len(A), len(A[0])
    _, colB = len(B), len(B[0])

    output = [[sum(A[i][k] * B[k][j] for k in range(colA)) for j in range(colB)] for i in range(rowA)]

    
    if colB == 1:
        output = [row[0] for row in output]
    
    return output


def identity_matrix(n: int) -> list:
    I = build_zero_matrix(n, n)
    for i in range(n):
        I[i][i] = 1
    return I

def add_matrix(A: list, B: list) -> list:
    assert len(A) == len(B) and len(A[0]) == len(B[0]), "A must = B"
    rowA, colA = len(A), len(A[0])
    for i in range(rowA):
        for j in range(colA):
            A[i][j] += B[i][j]
    return A

    
def check_2D_list(A: list) -> bool:
    return isinstance(A, list) and isinstance(A[0], list)

def LU_decomposition(A: list):
    n = len(A)   
    L = identity_matrix(n) 
    U = build_zero_matrix(n, n) 

    for i in range(n):
        for j in range(i, n):  # cal U[i][j]
            U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
        for j in range(i + 1, n):  # cal L[j][i]
            L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]
            
    return L, U

def forward_substitution(L: list, b: list) -> list:
    n = len(L)
    y = [0] * n 

    for i in range(n):
        sum_Ly = sum(L[i][j] * y[j] for j in range(i))  
        y[i] = (b[i] - sum_Ly) / L[i][i] 

    return y
    
def backward_substitution(U: list, y: list) -> list:
    n = len(U)
    x = [0] * n  

    for i in range(n - 1, -1, -1):  
        sum_Ux = sum(U[i][j] * x[j] for j in range(i + 1, n))  
        x[i] = (y[i] - sum_Ux) / U[i][i]  

    return x

def LU_solve_A_invert(A, b):
    L, U = LU_decomposition(A)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    
    return x

def mean_squared_error(y_pred: list, y: list) -> float:
    return sum((y_pred[i][0] - y[i][0]) ** 2 for i in range(len(y)))

    
    
if __name__ == "__main__":
    
    x = [1, 2, 3, 4, 5]  
    degree = 3

    A = polynomial_basis(x, degree)
    
    A_transpose = transpose(A)
    
    
