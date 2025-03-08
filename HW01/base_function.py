import numpy as np
from dataclasses import dataclass, field
from typing import Tuple
import argparse
import sys
import matplotlib.pyplot as plt


def load_data(file_path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(file_path, delimiter=',')
    return data[:, 0], data[:, 1]  
    
def polynomial_basis(x, degree):
    basis = np.array([x ** i for i in range(degree)])
    return basis.T

def plot_regression_curve(x, y, coefficients, degree, lambd):
    plt.scatter(x, y, label='Data points', color='blue')
    x_line = np.linspace(min(x) - 1, max(x) + 1, 100)
    y_line = np.dot(polynomial_basis(x_line, degree), coefficients).flatten()  # (n, 1) -> (n, )
    plt.plot(x_line, y_line, color='red', label='Fitted polynomial curve')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(f'Polynomial Regression (Degree {degree}) with λ={lambd}')
    plt.show()

def get_parser():
    parser = argparse.ArgumentParser(description="Load dataset and process it.")
    parser.add_argument("-f", "--file_path", type=str, default="/home/charlie/data/2025_ML/hw01/test.txt",
                        help="Path to the dataset file (CSV format)")
    parser.add_argument("-d", "--degree", type=int, default=1, help="Polynomial degree")
    parser.add_argument("-l", "--lambd", type=float, default=1.0, help="Regularization lambda")
    return parser

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        args = get_parser().parse_args()
    else:
        args = argparse.Namespace(file_path="/home/charlie/data/2025_ML/hw01/test.txt", degree=1, lambd=1.0)

    base = Base(file_path=args.file_path, 
                degree=args.degree, 
                lambd=args.lambd)

    # 測試輸出
    print("file_path:", base.file_path)
    print("x:", base.x)
    print("y:", base.y)
    print("lambda:", base.lambd)
    print("degree:", base.degree)
