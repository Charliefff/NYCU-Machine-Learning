from random_data_generator import univariate_gaussian

def sequential_estimator(m, s, threshold= 1e-1):
    print("Sequential Estimator")
    print("=================================")
    print()
    epoch, mean, M2, old_mean, old_var= 0, 0., 0, float('inf'), float('inf')
    
    while True:
        epoch += 1
        x = univariate_gaussian(m, s)
        delta = x - mean
        mean += delta / epoch
        delta2 = x - mean
        M2 += delta * delta2
        var = M2 / (epoch) if epoch > 1 else 0
        print(f"Add data point: {x}")
        print(f"Mean = {mean} Variance = {var}")
        if abs(mean - old_mean) < threshold and abs(var - old_var) < threshold:
            break
        old_mean, old_var = mean, var

        

if __name__ == "__main__":
    sequential_estimator(3, 5.0, 1e-2)
