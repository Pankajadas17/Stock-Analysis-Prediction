# Stock Analysis & Prediction

This is a simple Python project for cleaning, analyzing, and predicting stock prices using multiple regression models.  
It is primarily intended for learning and experimentation.

---

## Features
- Loads stock data from a CSV file.
- Detects and removes outliers using the IQR method.
- Trains several regression models:
  - Linear Regression
  - Ridge
  - Lasso
  - ElasticNet
  - Polynomial Regression (degree=2)
- Saves plots showing:
  - Outliers before and after removal
  - Actual vs Predicted scatter plots for each model
- Prints a comparison table with metrics including MSE, MAE, MBE, RMSE, R², and MLE.

---

## Files
- `stockprediction.py` : Main script that does all the analysis and prediction.
- `stock.csv` : Your dataset (must be placed in the same directory).
- Output files:
  - `outliers_before.png` and `outliers_after.png` : Outlier visualization.
  - `<model>_regression.png` : Prediction plots for each regression model.

---

## Requirements
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install the dependencies using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

## To run
python stockprediction.py
