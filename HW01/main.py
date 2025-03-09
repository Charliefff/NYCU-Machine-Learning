import numpy as np
import matplotlib.pyplot as plt
from base_function import polynomial_basis
import argparse
from LSE import rLSE
from typing import Tuple
from steepest_descent import linear_regression_via_steepest_descent

def plot_regression_curve(x, y, coefficients, degree, lambd):
    plt.scatter(x, y, label="Data points", color="blue")

    x_line = np.linspace(min(x) - 1, max(x) + 1, 100)    
    basis_matrix = polynomial_basis(x_line, degree)  # (n, degree)
    y_line = np.dot(basis_matrix, [[c] for c in coefficients])  # (n, 1)
    
    # 轉換回 1D list
    y_line = [yi[0] for yi in y_line]

    plt.plot(x_line, y_line, color="red", label="Fitted polynomial curve")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title(f"Polynomial Regression (Degree {degree}) with λ={lambd}")
    plt.show()
    
def get_parser():
    parser = argparse.ArgumentParser(description="Load dataset and process it.")
    parser.add_argument("-f", "--file_path", type=str, default="/home/charlie/data/2025_ML/HW01/test.txt",
                        help="Path to the dataset file (CSV format)")
    parser.add_argument("-d", "--degree", type=int, default=1, help="Polynomial degree")
    parser.add_argument("-l", "--lambd", type=float, default=1.0, help="Regularization lambda")
    parser.add_argument("-o", "--output", type=str, default="./picture/output.png", help="Output file path")
    return parser


def load_data(file_path) -> Tuple[list, list]:
    with open(file_path, 'r') as f:
        data = [list(map(float, line.strip().split(','))) for line in f.readlines()]
    
    x = [row[0] for row in data]
    y = [row[1] for row in data]
    return x, y

def run_LSE(x, y, degree, lambd, args):
    
    coefficients, _, total_error = rLSE(x, y, args.degree, args.lambd)
    print("Coefficients:\n", coefficients, "\n")
    print("Total Error:", total_error)
    plot_regression_curve(x, y, coefficients, args.degree, args.lambd)
    
def run_steepest_descent(x, y, degree, lambd, args):
    coefficients = linear_regression_via_steepest_descent(x, y, degree, lambd, epsilon=1e-10)
    print("Coefficients:\n", coefficients, "\n")
    plot_regression_curve(x, y, coefficients, degree, lambd)
    
def main():
    args = get_parser().parse_args()
    x, y = load_data(args.file_path)
    print("LSE")
    # run_LSE(x, y, args.degree, args.lambd, args)
    
    print("Steepest Descent")
    run_steepest_descent(x, y, args.degree, args.lambd, args)


    

if __name__ == "__main__":
    main()