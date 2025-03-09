from base_function import (polynomial_basis, 
                           transpose, 
                           matrix_multiplication,
                           mean_squared_error)

def compute_gradient(A, y, coefficients, lambd, epsilon):
    
    coefficients_reshape = [[c] for c in coefficients]  

    residuals = matrix_multiplication(A, coefficients_reshape)  # (n, 1)
    residuals = [[r] for r in residuals]  
    residuals = [[residuals[i][0] - y[i]] for i in range(len(y))]  # (n, 1)

    A_transpose = transpose(A)
    gradient_ls = matrix_multiplication(A_transpose, residuals)
    gradient_ls = [[g] for g in gradient_ls]  # (degree, 1)
    #  L1 norm
    gradient_l1_approx = [[c / (c**2 + epsilon) ** 0.5] for c in coefficients]  

    # Full gradient: 2 * A^T(Aw - y) + λ * gradient of L1 norm approximation
    gradient = [[2 * gradient_ls[i][0] + lambd * gradient_l1_approx[i][0]] for i in range(len(gradient_ls))]

    return gradient

def steepest_descent(A, y, degree, lambd, learning_rate=1e-4, tol=1e-10, max_iter=10000, epsilon=1e-10):
    coefficients_old = [0] * degree 
    coefficients_new = [0] * degree  

    for _ in range(max_iter):
        gradient = compute_gradient(A, y, coefficients_old, lambd, epsilon)

        coefficients_new = [coefficients_old[i] - learning_rate * gradient[i][0] for i in range(degree)]
        
        diff_norm = sum(abs(coefficients_new[i] - coefficients_old[i]) for i in range(degree))
        if diff_norm < tol:
            break

        coefficients_old = coefficients_new  

    return coefficients_new

def linear_regression_via_steepest_descent(x, y, degree, lambd, epsilon):
    A = polynomial_basis(x, degree)


    if isinstance(y[0], list):
        y = [row[0] for row in y]

    coefficients = steepest_descent(A, y, degree, lambd, epsilon)

    y_pred = matrix_multiplication(A, [[c] for c in coefficients])  # (n, 1)
    print(y_pred)
    y_pred = [[row] for row in y_pred]  # to 1D list
    y = [[row] for row in y]  # to 1D list
    total_error = mean_squared_error(y_pred, y)
    

    equation = "Fitting line: "
    for i, coef in enumerate(coefficients[::-1]):
        if i == 0:  # leading coefficient
            equation += f"{coef:.10f}X^{(degree - 1 - i)}"
        else:
            if i == (degree - 1):
                if coef > 0:
                    equation += f" + {coef:.10f}"
                else:
                    equation += f" - {abs(coef):.10f}"
            else:
                if coef > 0:
                    equation += f" + {coef:.10f}X^{(degree - 1 - i)}"
                else:
                    equation += f" - {abs(coef):.10f}X^{(degree - 1 - i)}"

    print("Steepest descent method:")
    print(equation)
    print(f"Total Error: {total_error:.10f}")

    return coefficients
