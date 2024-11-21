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

    $$z = \frac{x - \mu}{\sigma}$$

    Where:

    - $z$ is the standardized value.

    - $x$ is the original value.

    - $\mu$ is the mean of the training samples.

    - $\sigma$ is the standard deviation of the training samples.

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

## Modeling

Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

### Lasso Regression

#### Lasso Regression Formula

Lasso Regression, or Least Absolute Shrinkage and Selection Operator, is a type of linear regression that uses L1 regularization. The objective function for Lasso Regression is:

$$ \text{minimize} \left( \frac{1}{2n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^{p} | \beta_j | \right) $$

Where:

- $y_i$ is the actual value.

- $\hat{y}_i$ is the predicted value.

- $n$ is the number of data points.

- $\alpha$ is the regularization parameter.

- $\beta_j$ are the coefficients of the model.

- $p$ is the number of features.

    ```python
    param_grid = {'lasso__alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 20, 30, 40]}
    pipeline = make_pipeline(poly, lasso)
    lasso_grid_search = GridSearchCV(pipeline, param_grid, scoring='r2', cv=3)
    lasso_grid_search.fit(x_train_scaled, y_train)
    ```

#### Lasso Regression Description

Lasso Regression aims to minimize the sum of the squared residuals (the difference between the actual and predicted values) while also applying a penalty to the absolute values of the coefficients. This penalty term helps to shrink some coefficients to zero, effectively performing feature selection and reducing the complexity of the model.

The regularization parameter $\alpha$ controls the strength of the penalty. A higher value of $\alpha$ increases the penalty, leading to more coefficients being shrunk to zero. Conversely, a lower value of $\alpha$ reduces the penalty, making the model more similar to ordinary least squares regression.

### RandomForestRegressor

#### RandomForestRegressor Formula

RandomForestRegressor is an ensemble learning method for regression that operates by constructing multiple decision trees during training and outputting the mean prediction of the individual trees.

The formula for the prediction of a RandomForestRegressor is:

$$ \hat{y} = \frac{1}{T} \sum_{t=1}^{T} \hat{y}_t $$

Where:

- $\hat{y}$ is the final prediction.

- $T$ is the number of trees in the forest.

- $\hat{y}_t$ is the prediction of the $ t $-th tree.

    ```python
    param_grid = {
        'n_estimators': [50, 80, 100],
        'max_depth': [3, 5, 7]
    }

    rf = RandomForestRegressor()
    rf_grid_search = GridSearchCV(rf, param_grid, scoring='r2', cv=2)
    rf_grid_search.fit(x_train_scaled, y_train)

    y_pred_train = rf_grid_search.predict(x_train_scaled)
    y_pred_test = rf_grid_search.predict(x_test_scaled)
    ```

#### RandomForestRegressor Description

RandomForestRegressor works by creating a multitude of decision trees at training time and outputting the average prediction of the individual trees. It reduces overfitting by averaging multiple trees, which improves the model's generalization ability.

The key parameters of RandomForestRegressor include:

- **n_estimators**: The number of trees in the forest.
- **max_depth**: The maximum depth of the trees.
- **random_state**: Controls the randomness of the bootstrapping of the samples used when building trees.
- **n_jobs**: The number of jobs to run in parallel for both fit and predict.

### XGBoost

#### XGBoost Formula

XGBoost (Extreme Gradient Boosting) is an optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable. It implements machine learning algorithms under the Gradient Boosting framework.

The objective function for XGBoost in regression is:

$$ \text{minimize} \left( \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k) \right) $$

Where:

- $l(y_i, \hat{y}_i)$ is the loss function that measures the difference between the actual value $y_i$ and the predicted value $\hat{y}_i$.

- $\Omega(f_k)$ is the regularization term that penalizes the complexity of the model to prevent overfitting.

- $n$ is the number of data points.

- $K$ is the number of trees.

- $f_k$ represents the individual trees in the model.

    ```python
    model_xgb = XGBRegressor()

    model_xgb.fit(x_train_scaled, y_train)

    y_pred_train = model_xgb.predict(x_train_scaled)
    y_pred_test = model_xgb.predict(x_test_scaled)
    ```

#### XGBoost Description

XGBoost works by building an ensemble of decision trees, where each tree corrects the errors of the previous ones. The model is trained in an additive manner, meaning that new trees are added to the ensemble sequentially to improve the overall prediction accuracy.

The key components of the XGBoost model include:

- **Loss Function**: Measures how well the model's predictions match the actual values. Common loss functions for regression include Mean Squared Error (MSE) and Mean Absolute Error (MAE).
- **Regularization Term**: Helps to control the complexity of the model and prevent overfitting by penalizing large coefficients.
- **Learning Rate**: Controls the contribution of each tree to the final model. A lower learning rate requires more trees to be added to the model.
- **Tree Structure**: Defines the depth and number of leaves in each tree. Deeper trees can capture more complex patterns but are also more prone to overfitting.

#### Advantages and Disadvantages

| Model                  | Advantages                                                                 | Disadvantages                                                                 |
|------------------------|-----------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| **Lasso Regression**   | - Performs feature selection by shrinking coefficients to zero             | - Can underperform if there are highly correlated features                    |
|                        | - Simple and easy to interpret                                             | - Not suitable for complex non-linear relationships                           |
|                        | - Can handle multicollinearity                                             | - Sensitive to outliers                                                       |
| **RandomForestRegressor** | - Handles non-linear relationships well                                  | - Can be computationally expensive                                            |
|                        | - Reduces overfitting by averaging multiple trees                          | - Less interpretable compared to linear models                                |
|                        | - Handles missing values and maintains accuracy for large datasets         | - Requires tuning of hyperparameters                                          |
| **XGBoost**            | - High performance and accuracy                                            | - Can be complex to tune                                                      |
|                        | - Handles missing values and outliers well                                 | - Computationally intensive                                                   |
|                        | - Regularization to prevent overfitting                                    | - Less interpretable compared to simpler models                               |
|                        | - Supports parallel processing                                             |                                                                               |

#### Why Choose XGBoost as the Best Model?

I chose XGBoost as the best model because of its outstanding performance metrics:

- **R-squared (train)**: 0.9994696666165278
- **R-squared (test)**: 0.984976762423431

- **Mean Squared Error (train)**: 8.278906955938487e-06
- **Mean Squared Error (test)**: 0.00023525446120578586

These metrics indicate that the XGBoost model has excellent predictive power and generalizes well to unseen data.

## Evaluation

### R-squared (R²)

#### What is R-squared (R²) and How it Works?

R-squared (R²), or the coefficient of determination, measures the proportion of variance in the dependent variable that is explained by the independent variable(s) in a regression model. It ranges from 0 to 1, where 0 means the model explains none of the variability, and 1 means it explains all the variability. A higher R² value indicates a better fit of the model to the data, but it does not imply causation. For multiple regression, adjusted R² is often used to account for the number of predictors, as adding more variables can inflate R² artificially.

#### R-squared (R²) Formula

##### R-squared (R²) for Training Data

The formula for R-squared on the training data is:

$$ R^2_{\text{train}} = 1 - \frac{SS_{\text{res, train}}}{SS_{\text{tot, train}}} $$

Where:

- $SS_{\text{res, train}}$ is the sum of squares of residuals for the training data.

- $SS_{\text{tot, train}}$ is the total sum of squares for the training data.

##### R-squared (R²) for Test Data

The formula for R-squared on the test data is:

$$ R^2_{\text{test}} = 1 - \frac{SS_{\text{res, test}}}{SS_{\text{tot, test}}} $$

Where:

- $SS_{\text{res, test}}$ is the sum of squares of residuals for the test data.

- $SS_{\text{tot, test}}$ is the total sum of squares for the test data.

### Mean Squared Error (MSE)

#### What is Mean Squared Error (MSE) and How it Works?

Mean Squared Error (MSE) measures the average squared difference between actual and predicted values in a regression model. It quantifies how well the model predicts the target variable, with lower values indicating better performance. By squaring the differences, MSE penalizes larger errors more heavily, making it sensitive to outliers. It is always non-negative, with an MSE of zero representing a perfect fit where predictions exactly match the actual values.

#### Mean Squared Error (MSE) Formula

$$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

Where:

- $y_i$ is the actual value.

- $\hat{y}_i$ is the predicted value.

- $n$ is the number of data points.

### Explaining Project Results Based on Evaluation Metrics

The evaluation metrics used in this project are R-squared (R²) and Mean Squared Error (MSE). These metrics help to assess the performance and accuracy of the machine learning models used for predicting gold prices.

- **R-squared (R²):** This metric indicates the proportion of variance in the dependent variable that is explained by the independent variables in the model. An R² value closer to 1 signifies a better fit of the model to the data. In this project, the XGBoost model achieved an R² of 0.9995 on the training data and 0.9850 on the test data, indicating excellent predictive power and generalization to unseen data.

- **Mean Squared Error (MSE):** This metric measures the average squared difference between the actual and predicted values. A lower MSE indicates better model performance. The XGBoost model achieved an MSE of 8.28e-06 on the training data and 0.00024 on the test data, demonstrating its high accuracy in predicting gold prices.

These results highlight the effectiveness of the XGBoost model in capturing the underlying patterns in the data and making accurate predictions, making it a reliable tool for investors and market analysts.
