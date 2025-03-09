from base_function import matrix_multiplication, transpose, inverse_matrix, compute_hessian_and_gradient, build_zero_matrix
import numpy as np

def newton_method(A, y, degree, tol=1e-10, max_iter=10000):
    coefficients_old = build_zero_matrix(degree, degree)
    coefficients_new = build_zero_matrix(degree, degree)

    for i in range(max_iter):
        gradient, hessian = compute_hessian_and_gradient(A, y, coefficients_old)
        hessian_inv = inverse_matrix(hessian)
        delta_coefficients = matrix_multiplication(hessian_inv, gradient).flatten()  # (degree, 1) -> (degree, )
        coefficients_new = coefficients_old - delta_coefficients

        if np.linalg.norm(abs(coefficients_new - coefficients_old)) < tol:
            break

        coefficients_old = coefficients_new

    return coefficients_new