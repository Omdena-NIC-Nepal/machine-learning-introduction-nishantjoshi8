def calculate_mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

def calculate_r2(y_true, y_pred):
    ss_total = ((y_true - y_true.mean()) ** 2).sum()
    ss_residual = ((y_true - y_pred) ** 2).sum()
    return 1 - (ss_residual / ss_total)

def plot_residuals(y_true, y_pred):
    import matplotlib.pyplot as plt
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values')
    plt.show()

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = calculate_mse(y_test, y_pred)
    r2 = calculate_r2(y_test, y_pred)
    plot_residuals(y_test, y_pred)
    return mse, r2