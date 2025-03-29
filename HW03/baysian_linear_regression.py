import numpy as np
from random_data_generator import polynomial_basis

def initialize_prior(b, dimension):
    cov_init = np.identity(dimension) / b       # 先驗協方差
    lambda_init = np.linalg.inv(cov_init)       # 先驗精確度矩陣
    mean_init = np.zeros((dimension, 1))        # 先驗平均
    return lambda_init, mean_init

def design_matrix_func(n):
    return lambda x: np.array([x**i for i in range(n)]).reshape(1, -1)

def update_posterior(phi_x, a, prior_lambda, prior_mean, y):

    post_lambda = prior_lambda + (1 / a) * (phi_x.T @ phi_x)
    post_cov = np.linalg.inv(post_lambda)
    post_mean = post_cov @ ((1 / a) * phi_x.T * y + prior_lambda @ prior_mean)
    return post_lambda, post_cov, post_mean

def predictive_distribution(phi_x, post_cov, post_mean, a):

    mean_pred = (phi_x @ post_mean).item(0)
    var_pred = a + (phi_x @ post_cov @ phi_x.T).item(0)
    return mean_pred, var_pred

def bayes_linear_regression(b, n, a, w, threshold=1e-6):
    
    prior_lambda_, prior_mean = initialize_prior(b, n)
    phi = design_matrix_func(n)
    
    
    x_storage, y_storage = [], []
    prev_var_pred = None  
    epochs = 0

    while epochs < 1000:
        x_val, y_val = polynomial_basis(n, a, w)
        x_storage.append(x_val)
        y_storage.append(y_val)

        
        phi_x = phi(x_val)
        
        posterior_lambda_, posterior_cov, posterior_mean = update_posterior(
            phi_x, a, prior_lambda_, prior_mean, y_val
        )
        

        mean_pred, var_pred = predictive_distribution(phi_x, posterior_cov, posterior_mean, a)
        
        if prev_var_pred is not None:
            if abs(var_pred - prev_var_pred) < threshold:
                break
        prev_var_pred = var_pred
        
        prior_lambda_ = posterior_lambda_
        prior_mean = posterior_mean
        
        epochs += 1
        
        print("Add data point ({}, {}):".format(x_val, y_val))
        print()
        print("Posterior mean:")
        print(posterior_mean)
        print()
        print("Posterior variance:")
        print(posterior_cov)
        print()
        print("Predictive distribution ~ N({:.5f}, {:.5f})".format(mean_pred, var_pred))
        print()

    print("Total number of iterations: ", epochs)

if __name__ == "__main__":
    b = 1
    n = 4
    a = 1
    w = [1, 2, 3, 4]
    bayes_linear_regression(b, n, a, w)
