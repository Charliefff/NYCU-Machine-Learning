from base_function import (
    transpose, 
    matrix_multiplication
)

def compute_gradient(A, y, coefficients, lambd, epsilon):

    coeff_reshape = [[c] for c in coefficients]  

   
    residuals = matrix_multiplication(A, coeff_reshape)
    if isinstance(residuals[0], float):  
        residuals = [[val] for val in residuals]

    residuals = [[residuals[i][0] - y[i]] for i in range(len(y))]  # (n,1)

    A_T = transpose(A)   
    gradient_ls = matrix_multiplication(A_T, residuals)  # (degree,1)

    # (3) L1 norm 
    gradient_l1_approx = [[c / ((c**2 + epsilon)**0.5)] for c in coefficients]  

    gradient = []
    for i in range(len(gradient_ls)):
        val = 2 * gradient_ls[i][0] + lambd * gradient_l1_approx[i][0]
        gradient.append([val])  # (degree,1)

    return gradient  # (degree,1)


def steepest_descent(A, y, degree, lambd, learning_rate=1e-4,
                     tol=1e-10, max_iter=10000, epsilon=1e-10):

    coefficients_old = [0.0] * degree
    coefficients_new = [0.0] * degree

    for _ in range(max_iter):

        gradient = compute_gradient(A, y, coefficients_old, lambd, epsilon)

        coefficients_new = [
            coefficients_old[i] - learning_rate * gradient[i][0]
            for i in range(degree)
        ]

        diff_norm = sum(abs(coefficients_new[i] - coefficients_old[i]) 
                        for i in range(degree))
        if diff_norm < tol:
            break

        coefficients_old = coefficients_new

    return coefficients_new



