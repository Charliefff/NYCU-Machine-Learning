from base_function import load_data, polynomial_basis, plot_regression_curve, get_parser
import numpy as np
import matplotlib.pyplot as plt

def rLSE(x, y, degree, lambd):
    
    A = polynomial_basis(x, degree) 
    A_transpose = A.T  # 轉置
    A_transpose_A = A_transpose @ A 
    y_reshape = y.reshape(-1, 1)  
    A_transpose_y = A_transpose @ y_reshape 
                  
    reg_matrix = A_transpose_A + lambd * np.eye(A_transpose_A.shape[0])
    
    coefficients = np.linalg.solve(reg_matrix, A_transpose_y)

    y_pred = A @ coefficients
    total_error = np.sum((y_pred - y_reshape) ** 2)
    
    return coefficients, y_pred, total_error

def plot_regression_curve(x, y, coefficients, degree, lambd):
    
    plt.figure(figsize=(15, 6))  
    plt.scatter(x, y, label="Data points", color="blue")  
    
    # 生成 x 軸範圍
    x_line = np.linspace(min(x) - 1, max(x) + 1, 100)  
    A_line = polynomial_basis(x_line, degree)
    y_line = A_line @ coefficients  
    
    plt.plot(x_line, y_line, color="red", label=f"Fitted polynomial (Degree {degree})")
    
    # 設定圖表標籤
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title(f"Polynomial Regression (Degree {degree}) with λ={lambd}")
    plt.savefig('/home/charlie/data/2025_ML/hw01/picture/output.png')
    
if __name__ == "__main__":

    x, y = load_data('/home/charlie/data/2025_ML/hw01/test.txt')

    coefficients, y_pred, total_error = rLSE(x, y, degree=2, lambd=0)


    
    print(coefficients, "\n")
    print(total_error)
    plot_regression_curve(x, y, coefficients, degree=2, lambd=0)
    

