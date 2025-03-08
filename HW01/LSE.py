from base_function import (polynomial_basis, 
                           transpose, 
                           matrix_multiplication, 
                           identity_matrix, 
                           add_matrix,
                           LU_solve_A_invert)

def rLSE(x: list, y: list, degree, lambd):
    
    A = polynomial_basis(x, degree) 
    A_transpose = transpose(A)
    A_transpose_A = matrix_multiplication(A_transpose, A)
    y_reshape = [[yi] for yi in y]  
    
    A_transpose_y = matrix_multiplication(A_transpose, y_reshape)

    identity = identity_matrix(len(A_transpose_A))
    scaled_identity = [[lambd * identity[i][j] for j in range(len(identity[0]))] for i in range(len(identity))]
    
    reg_matrix = add_matrix(A_transpose_A, scaled_identity)
    
    coefficients = LU_solve_A_invert(reg_matrix, A_transpose_y)  # 解 Ax = b
    
    y_pred = matrix_multiplication(A, coefficients)
    y_pred = [[y_p] for y_p in y_pred]
    
    total_error = sum((y_pred[i][0] - y_reshape[i][0]) ** 2 for i in range(len(y_reshape)))  # 手動平方誤差

    return coefficients, y_pred, total_error
