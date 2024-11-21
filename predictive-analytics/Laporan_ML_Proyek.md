
# Laporan Proyek Machine Learning - Ridwan Halim

## Project Domain

This project aims to predict gold prices using machine learning techniques. Gold prices are known to be highly volatile and influenced by various economic and political factors. Therefore, having an accurate prediction model can help investors and market analysts make better decisions.

**Why and How This Problem Should Be Solved:**

The issue of gold price fluctuation needs to be addressed because it can affect economic stability and investment decisions. By using machine learning models, we can analyze historical data and the factors influencing gold prices to make more accurate predictions. This will help reduce risks and increase profits for investors.

**Related Research or References:**

- DigitalOcean. "Using `StandardScaler()` Function to Standardize Python Data" Available at [DigitalOcean](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

- scikit-learn. "StandardScaler" Available at [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

- Shah, I., & Pachanekar, R. (2021). Gold Price Prediction Using Machine Learning In Python. Retrieved from [QuantInsti](https://blog.quantinsti.com/gold-price-prediction-using-machine-learning-python/).

- Youssef, M. (2021). Gold Price Prediction Using Random Forest. Retrieved from [GitHub](https://github.com/MYoussef885/Gold_Price_Prediction).
- Pendry, P. S. (2021). Gold Price Prediction using Random Forest. Retrieved from [GitHub](https://github.com/pavansaipendry/Gold-Price-Prediction).

- Ben Jabeur, S., Mefteh-Wali, S., & Viviani, J. L. (2021). Forecasting gold price with the XGBoost algorithm and SHAP interaction values. *Annals of Operations Research*, 334, 679-699. Retrieved from [Springer](https://link.springer.com/article/10.1007/s10479-021-04187-w).
- Theja, A. (2021). Gold Price Prediction Using XGBoost. Retrieved from [GitHub](https://github.com/abhijantheja/gold_predictor).

- GeeksforGeeks. "Gold Price Prediction using Machine Learning." Available at [GeeksforGeeks](https://www.geeksforgeeks.org/gold-price-prediction-using-machine-learning/).

- Analytics Vidhya. "Building A Gold Price Prediction Model Using Machine Learning" Available at [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2021/07/building-a-gold-price-prediction-model-using-machine-learning/).

## Business Understanding

### Problem Statements

- The gold price is highly volatile and influenced by various economic and political factors, making it challenging for investors to predict future prices accurately.

- The lack of accurate prediction models can lead to poor investment decisions, resulting in financial losses.

### Goals

- Develop a machine learning model to predict gold prices accurately.

- Provide investors and market analysts with a reliable tool to make informed decisions, thereby reducing financial risks and increasing potential profits.

### Solution statements

01. Solution 1: Implement Multiple Algorithms

    - Use multiple machine learning algorithms such as Linear Regression, Random Forest, and XGBoost to predict gold prices.

    - Compare the performance of these algorithms to identify the most accurate model.

    - Evaluation Metrics: R-squared, Mean Squared Error (MSE).

02. Solution 2: Improve Baseline Model with Hyperparameter Tuning

    - Start with a baseline model, such as Linear Regression.

    - Perform hyperparameter tuning to optimize the model's performance.

    - Evaluation Metrics: R-squared, Mean Squared Error (MSE).

03. Solution 3: Ensemble Methods

    - Combine the predictions of multiple models using ensemble methods like Bagging and Boosting to improve accuracy.

    - Evaluation Metrics: R-squared, Mean Squared Error (MSE).

## Data Understanding

The dataset for this project is taken from GeeksforGeeks and has been modified to be easy to use for beginners learning machine learning, especially in time series. The original source of the gold price data is from Yahoo Finance. The dataset can be downloaded from the following source: [GitHub Repository](https://github.com/ridwaanhall/applied-machine-learning/raw/refs/heads/main/predictive-analytics/data/gold_price_data.csv).

Next, let's describe all the variables or features in the data.

### Variables in the Gold Price Dataset

| Variable | Note                                      | Example Value |
|----------|-------------------------------------------|---------------|
| Date     | The date of the recorded data             | 2024-11-21    |
| SPX      | S&P 500 index value                       | 4500.25       |
| GLD      | Gold price                                | 1800.50       |
| USO      | Crude oil price (United States Oil Fund)  | 55.75         |
| SLV      | Silver price (iShares Silver Trust)       | 24.30         |
| EUR/USD  | Exchange rate between Euro and US Dollar  | 1.12          |

- The dataset contains 2290 entries for each variable.
- The `Date` column should be converted from `object` to `datetime` for better analysis.
- The `SPX`, `GLD`, `USO`, `SLV`, and `EUR/USD` columns are all of type `float64`.
- No missing values are present in the dataset.
- The `GLD` column (gold price) has a mean of 122.73 and a standard deviation of 23.28.
- The `USO` column (crude oil price) has a wide range, with a minimum of 7.96 and a maximum of 117.48, indicating high volatility.
- The `EUR/USD` exchange rate ranges from 1.039 to 1.599, with a mean of 1.284.

### EDA

- Data Wrangling: This is the process of cleaning and transforming raw data into a format that is more appropriate for analysis. It includes handling missing values, correcting data types, and removing duplicates.
- Distribution of Columns: Analyzing the distribution of data in each column helps to understand the spread and central tendency of the data. This can be done using histograms, bar charts, or summary statistics like mean, median, and standard deviation.
- Plotting Boxplot to Visualize the Outliers: A boxplot is a graphical representation that shows the distribution of data based on a five-number summary: minimum, first quartile (Q1), median, third quartile (Q3), and maximum. It helps to identify outliers and understand the spread and skewness of the data.

## Data Preparation

1. **Splitting the Data:**

    ```python
    X = gold_price.drop(columns=['Date', 'EUR/USD'])
    y = gold_price['EUR/USD']

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

    - The `Date` and target variable columns are dropped, and the remaining data is stored in the `X` variable as independent variables. The target variable is stored in the `y` variable.
    - The dataset is split into training and testing sets in an 80:20 ratio.

2. **Scaling the Data:**

    The formula for standardizing data using the `StandardScaler` is:

    $$ z = \frac{x - \mu}{\sigma} $$

    Where:
    - \( z \) is the standardized value.
    - \( x \) is the original value.
    - \( \mu \) is the mean of the training samples.
    - \( \sigma \) is the standard deviation of the training samples.

    This formula transforms the data to have a mean of 0 and a standard deviation of 1 [Function to Standardize Python Data](https://www.digitalocean.com/community/tutorials/standardscaler-function-in-python).

    ```python
    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    ```

     - The `StandardScaler` is used to standardize the data, transforming it to have a mean of 0 and a standard deviation of 1.
     - Fit the StandardScaler on the training dataset and transform both training and testing datasets.
     - This step is crucial for ensuring that all features contribute equally to the model.

### Reasons for Data Preparation Steps

- **Splitting the Data:** Separates the data into training and testing sets, allowing for model evaluation.
- **Scaling the Data:** Standardizes the data, ensuring that all features contribute equally to the model and improving the model's performance.
