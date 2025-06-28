import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline

# load the data
df = pd.read_csv("/workspaces/Stock-Analysis-Prediction/stock.csv").dropna()

# select numerical columns for outlier checks
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# visualize outliers
def plot_outliers(data, cols, title, filename):
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(cols):
        plt.subplot(1, len(cols), i + 1)
        sns.boxplot(y=data[col], color='skyblue')
        sns.stripplot(y=data[col], color='red', alpha=0.4, size=3)
        plt.title(f'{title} - {col}')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"Saved outlier plot: {filename}")

plot_outliers(df, num_cols, "Before", "outliers_before.png")

# remove outliers using IQR method
def remove_outliers(data, cols):
    for col in cols:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        data = data[(data[col] >= lower) & (data[col] <= upper)]
    return data

df = remove_outliers(df, num_cols)

plot_outliers(df, num_cols, "After", "outliers_after.png")

# setup features and target
target = 'Close'
X = df.drop(columns=[target, 'Date'])
y = df[target]

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# utility funcs
def eval_model(model, X, y, name):
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"{name:<30} | MSE: {mse:.4f} | R²: {r2:.4f}")
    return mse, r2, preds

def plot_preds(y_true, y_pred, title, filename):
    plt.figure(figsize=(6,5))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"Saved prediction plot: {filename}")

# models to try
results = {}

models = [
    ("Linear Regression", LinearRegression(), "linear_regression.png"),
    ("Ridge Regression", Ridge(alpha=1.0), "ridge_regression.png"),
    ("Lasso Regression", Lasso(alpha=0.1), "lasso_regression.png"),
    ("ElasticNet Regression", ElasticNet(alpha=0.1, l1_ratio=0.5), "elasticnet_regression.png")
]

for name, model, file in models:
    model.fit(X_train_scaled, y_train)
    mse, r2, preds = eval_model(model, X_test_scaled, y_test, name)
    results[name] = (y_test, preds)
    plot_preds(y_test, preds, name, file)

# polynomial regression
poly_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
poly_model.fit(X_train_scaled, y_train)
poly_preds = poly_model.predict(X_test_scaled)
results["Polynomial Regression (deg=2)"] = (y_test, poly_preds)
plot_preds(y_test, poly_preds, "Polynomial Regression (Degree 2)", "polynomial_regression.png")

# extra error metrics
def mbe(y_true, y_pred):
    return np.mean(y_pred - y_true)

def mle(y_true, y_pred):
    res = y_true - y_pred
    n = len(res)
    sigma2 = np.var(res)
    return 0.5 * n * np.log(2 * np.pi * sigma2) + np.sum(res**2) / (2 * sigma2)

# final summary
print("\nModel Comparison:")
print(f"{'Model':35} | {'MSE':>10} | {'MAE':>10} | {'MBE':>10} | {'RMSE':>10} | {'R²':>6} | {'MLE':>10}")
print("-" * 100)

for name, (y_true, y_pred) in results.items():
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mb_err = mbe(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mle_val = mle(y_true, y_pred)
    print(f"{name:35} | {mse:10.4f} | {mae:10.4f} | {mb_err:10.4f} | {rmse:10.4f} | {r2:6.4f} | {mle_val:10.4f}")
