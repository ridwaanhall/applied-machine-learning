# Machine Learning Project Report - Ridwan Halim

[![wakatime](https://wakatime.com/badge/user/018b799e-de53-4f7a-bb65-edc2df9f26d8/project/45c71873-666f-4140-a133-d302f409bd33.svg)](https://wakatime.com/badge/user/018b799e-de53-4f7a-bb65-edc2df9f26d8/project/45c71873-666f-4140-a133-d302f409bd33)

## Project Domain

This project aims to predict gold prices using machine learning techniques. Gold prices are known to be highly volatile and influenced by various economic and political factors. Therefore, having an accurate prediction model can help investors and market analysts make better decisions.

**Why and How This Problem Should Be Solved:**

The issue of gold price fluctuation needs to be addressed because it can affect economic stability and investment decisions. By using machine learning models, we can analyze historical data and the factors influencing gold prices to make more accurate predictions. This will help reduce risks and increase profits for investors.

## Business Understanding

### Problem Statements

- Volatility of Gold Prices: The price of gold is highly volatile and changes rapidly over time. This unpredictability makes it challenging for investors and analysts to make informed decisions about buying or selling gold.

- Data Complexity: The dataset used for predicting gold prices includes multiple variables such as SPX, GLD, USO, SLV, and EUR/USD. Handling and preprocessing this complex data to make it suitable for model training is a significant challenge.

### Goals

- Accurate Prediction: The primary goal of the project is to develop a machine learning model that can accurately predict the price of gold per unit. This involves analyzing historical data and identifying patterns that can help forecast future prices.

- Model Deployment: Another goal is to deploy the trained model so that it can be used in real-time to predict gold prices. This includes saving the model using tools like Pickle and integrating it with a web framework for live predictions.

### Solution statements

- Data Preprocessing: This includes handling missing values, normalizing data, and removing outliers to ensure the dataset is clean and suitable for model training.

- Model Development and Deployment: Various machine learning models such as Lasso Regression, RandomForestRegressor, and XGBoost are used to train on the dataset. The best-performing model is then deployed using Pickle for real-time predictions.

## Data Understanding

The dataset for this project is taken from GeeksforGeeks and has been modified to be easy to use for beginners learning machine learning, especially in time series. The original source of the gold price data is from Yahoo Finance. The dataset can be downloaded from the following source: [GitHub Repository](https://github.com/ridwaanhall/applied-machine-learning/raw/refs/heads/main/predictive-analytics/data/gold_price_data.csv).

### Data Summary

- Rows: 2290
- Columns: 6

### Data Condition

- Missing Values: There are no missing values in any of the columns.
- Duplicate Data: There are no duplicate data.
- Outliers: The `USO` column has outliers, which were handled by capping the values at the 95th and 5th percentiles.

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

#### Data Wrangling

![Data Wrangling](images/data-wrangling.png)

This graph does not provide clear insights into the changes in the price of gold due to its noisy appearance.

To better observe the trend, we need to smooth the data.

![Data Wrangling Fixed](images/data-wrangling-fixed.png)

The graph is now less noisy, allowing us to better analyze the trend in the change of gold prices.

#### Distribution of Columns

![Distribution of Columns](images/histogram-distribution.png)

The data distribution looks good. However, we must calculate skewness along the index axis.

```python
print(dataset.drop("Date", axis=1).skew(axis=0, skipna=True))
```

the output:

Skewness of each column:
SPX            0.300362
GLD            0.334138
USO            1.699331
EUR/USD       -0.005292
price_trend   -0.029588
dtype: float64

Column `USO` has the highest skewness of 0.98, so a square root transformation will be applied to this column to reduce its skewness to 0 in the visualize the outliers section.

#### Visualize the Outliers

![Distribution of Columns](images/histogram-boxplot.png)

It is clear that `USO` has outliers.

## Data Preparation

### Drop SLV

Drop the SLV column since the GLD column also shows a significant correlation with our target variable.

```python
gold_price.drop("SLV", axis=1, inplace=True)
```

### Remove Outliers

Based on EDA, there are outliers, Here is code to remove outliers:

```python
def remove_outliers(column):
    upper_limit = column.quantile(0.95)
    lower_limit = column.quantile(0.05)

    column = column.clip(lower=lower_limit, upper=upper_limit)

    return column

gold_price[['SPX', 'GLD', 'USO', 'EUR/USD']] = \
    gold_price[['SPX', 'GLD', 'USO', 'EUR/USD']].apply(remove_outliers)
```

### Splitting the Data

Separates the data into training and testing sets, allowing for model evaluation.

```python
X = gold_price.drop(columns=['Date', 'EUR/USD'])
y = gold_price['EUR/USD']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

- The `Date` and target variable columns are dropped, and the remaining data is stored in the `X` variable as independent variables. The target variable is stored in the `y` variable.
- The dataset is split into training and testing sets in an 80:20 ratio.

### Scaling the Data

Standardizes the data, ensuring that all features contribute equally to the model and improving the model's performance.

The formula for standardizing data using the `StandardScaler` is:

$$z = \frac{x - \mu}{\sigma}$$

Where:

- $z$ is the standardized value.

- $x$ is the original value.

- $\mu$ is the mean of the training samples.

- $\sigma$ is the standard deviation of the training samples.

This formula transforms the data to have a mean of 0 and a standard deviation of 1 Function to Standardize Python Data.[1] [2]

```python
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

imputer = SimpleImputer(strategy='mean')

# Fit and transform the imputer on the scaled training data
x_train_scaled = imputer.fit_transform(x_train_scaled)

# Transform the scaled test data using the trained imputer
x_test_scaled = imputer.transform(x_test_scaled)
```

- The `StandardScaler` is used to standardize the data, transforming it to have a mean of 0 and a standard deviation of 1.
- Fit the StandardScaler on the training dataset and transform both training and testing datasets.
- This step is crucial for ensuring that all features contribute equally to the model.

## Model Development

### Lasso Regression

#### Lasso Regression Formula

Lasso Regression, or Least Absolute Shrinkage and Selection Operator, is a type of linear regression that uses L1 regularization[3]. The objective function for Lasso Regression is:

$$ minimize \left( \frac{1}{2n} \sum_{i=1}^{n} (y_i - \hat{y}i)^2 + \alpha \sum_{j=1}^{p} |\beta_j| \right) $$

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

#### Best Parameter Value from Lasso Regression

```plaintext
Best parameter values:  {'lasso__alpha': 0.0001}
Best score:  0.9675368417416342
```

### RandomForestRegressor

#### RandomForestRegressor Formula

RandomForestRegressor is an ensemble learning method for regression that operates by constructing multiple decision trees during training and outputting the mean prediction of the individual trees[4].

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

The key parameters of RandomForestRegressor include[4]:

- **n_estimators**: The number of trees in the forest.
- **max_depth**: The maximum depth of the trees.
- **random_state**: Controls the randomness of the bootstrapping of the samples used when building trees.
- **n_jobs**: The number of jobs to run in parallel for both fit and predict.

#### Best Parameter Value from RandomForestRegressor

```plaintext
Best parameter values:  {'max_depth': 7, 'n_estimators': 50}
Best score:  0.9771392110600441
```

### XGBoost

#### XGBoost Formula

XGBoost (Extreme Gradient Boosting) is an optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable. It implements machine learning algorithms under the Gradient Boosting framework.

The objective function for XGBoost in regression is:

$$ minimize \left( \sum_{i=1}^{n} l(y_i, \hat{y}i) + \sum{k=1}^{K} \Omega(f_k) \right) $$

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

#### Best Parameter Value from XGBoost

```plaintext
Best parameter values:  Best Parameters:  {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 300, 'subsample': 0.8}
```

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

- R-squared (train):  0.9997376396433267
- R-squared (test):  0.9852577428747992
- Mean Squared Error (train):  4.095644455953784e-06
- Mean Squared Error (test):  0.0002308544838800552

These metrics indicate that the XGBoost model has excellent predictive power and generalizes well to unseen data.

## Evaluation

### R-squared (R²)

#### What is R-squared (R²) and How it Works?

R-squared (R²), or the coefficient of determination, measures the proportion of variance in the dependent variable that is explained by the independent variable(s) in a regression model. It ranges from 0 to 1, where 0 means the model explains none of the variability, and 1 means it explains all the variability. A higher R² value indicates a better fit of the model to the data, but it does not imply causation. For multiple regression, adjusted R² is often used to account for the number of predictors, as adding more variables can inflate R² artificially[5].

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

#### Lasso Regression Evaluation

The evaluation metrics for the Lasso Regression model are as follows:

- R-squared (train):  0.9684811019451044
- R-squared (test):  0.9609210669970879
- Mean Squared Error (train):  0.0004920339403145819
- Mean Squared Error (test):  0.0006119515371597247

These metrics indicate that the Lasso Regression model performs well on both the training and test data, with a high R-squared value and low Mean Squared Error. However, it is slightly less accurate compared to the XGBoost model and RandomForestRegressor model.

##### Impact of Lasso Regression Model on `Business Understanding` based on Evaluation

Lasso Regression addresses the volatility of gold prices by providing a stable and interpretable model, performing feature selection by shrinking some coefficients to zero, which reduces complexity and aids in handling and preprocessing complex data. This makes the model more robust, less prone to overfitting, and ensures more reliable predictions. It supports the goal of accurate prediction by reducing overfitting and improving generalization, leading to better-informed investment decisions. Lasso Regression models are simple and efficient, making them easier to deploy in real-time applications with less computational power. It emphasizes data preprocessing, ensuring a clean dataset for better performance.

#### RandomForestRegressor Evaluation

The evaluation metrics for the RandomForestRegressor model are as follows:

- R-squared (train): 0.9853448468612149
- R-squared (test): 0.9753603891969073
- Mean Squared Error (train): 0.0002287780725148166
- Mean Squared Error (test): 0.00038584082387424333

These metrics indicate that the RandomForestRegressor model performs well on both the training and test data, with a high R-squared value and low Mean Squared Error. However, it is slightly less accurate compared to the XGBoost model.

##### Impact of RandomForestRegressor Model on `Business Understanding` based on Evaluation

RandomForestRegressor model addresses the challenge of predicting highly volatile gold prices by providing a robust and flexible solution that captures complex patterns in the data and effectively handles datasets with multiple variables, making it suitable for preprocessing and analyzing intricate data relationships. With high R-squared values (0.985 for training and 0.975 for testing) and low Mean Squared Errors (0.000228 for training and 0.000385 for testing), the model accurately predicts gold prices, supporting the primary goal of making well-informed investment decisions. Its robustness and flexibility make it suitable for real-time deployment, ensuring reliable predictions crucial for live applications. The model's ensemble nature reduces the risk of overfitting and improves generalization, leading to accurate predictions over time, while its capability to manage complex datasets simplifies preprocessing, resulting in more efficient and effective data handling.

#### XGBoost Evaluation

The evaluation metrics for the XGBoost model are as follows:

- R-squared (train):  0.9997376396433267
- R-squared (test):  0.9852577428747992
- Mean Squared Error (train):  4.095644455953784e-06
- Mean Squared Error (test):  0.0002308544838800552

These metrics indicate that the XGBoost model performs well on both the training and test data, with a high R-squared value and low Mean Squared Error.

##### Impact of XGBoost Model on `Business Understanding` based on Evaluation

The XGBoost model effectively addresses the challenge of predicting highly volatile gold prices by capturing complex patterns in the data and handling datasets with multiple variables. With high R-squared values (0.9997 for training and 0.9853 for testing) and low Mean Squared Errors (4.10e-6 for training and 0.000231 for testing), the model demonstrates exceptional accuracy in predicting gold prices. These metrics highlight the model's ability to support informed investment decisions by providing precise and reliable predictions. Its ensemble nature reduces overfitting, enhances generalization, and ensures robust performance over time. Furthermore, the model's capability to handle intricate datasets simplifies preprocessing, making it well-suited for real-time deployment and live applications where consistent reliability is paramount.

**Related Research or References:**

[1] DigitalOcean. "Using `StandardScaler()` Function to Standardize Python Data" Available at [DigitalOcean](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

[2] scikit-learn. "StandardScaler" Available at [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

[3] GeeksforGeeks. "Gold Price Prediction using Machine Learning." Available at [GeeksforGeeks](https://www.geeksforgeeks.org/gold-price-prediction-using-machine-learning/).

[4] scikit-learn. "RandomForestRegressor" Available at [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

[5] GeeksforGeeks. "R Squared | Coefficient of Determination" Available at [GeeksforGeeks](https://www.geeksforgeeks.org/r-squared/)

[6] GeeksforGeeks "Mean Squared Error" Available at [GeeksforGeeks](https://www.geeksforgeeks.org/mean-squared-error/)

[7] scikit-learn. "RandomForestRegressor" Available at [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
