# Loan Default Prediction Using Big Data Tools and Techniques

This project leverages big data technologies and machine learning techniques to predict loan defaults, enabling financial institutions to assess risk more effectively and make informed lending decisions.

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Technologies Used](#technologies-used)
4. [Key Features](#key-features)
5. [Data Operations](#data-operations)
6. [Modeling](#modeling)
7. [Results](#results)
8. [Future Scope](#future-scope)
9. [References](#references)

---

## Overview

Loan default prediction is a critical task for financial institutions. This project explores borrower characteristics and loan features to identify factors influencing loan default. Using a comprehensive data pipeline, we preprocess, visualize, and model the data to gain actionable insights.

## Dataset

- **Source**: [Home Credit Default Risk Dataset](https://www.kaggle.com/c/home-credit-default-risk/data)
- **Size**: 288,161 rows, 58 columns
- **Features**: Loan amount, borrower credit score, annual income, employment status, debt-to-income ratio, payment history, and more.
- **Target**: `0` for loans repaid on time, `1` for loans defaulted (class imbalance: 96.9% repaid, 3.1% defaulted).

## Technologies Used

- **Big Data Frameworks**: Apache Spark, PySpark, Apache Hive
- **Programming Language**: Python
- **Libraries**: PySpark MLlib, Matplotlib, Seaborn
- **Machine Learning Models**: ElasticNet Regression, K-Means Clustering

## Key Features

- **Data Cleaning**: Handling missing values, removing duplicates, outlier capping.
- **Exploratory Data Analysis (EDA)**: Comprehensive visualization of data relationships.
- **Clustering**: Segmenting borrowers into groups using K-Means.
- **Predictive Modeling**: ElasticNet regression for predicting loan defaults.

## Data Operations

1. **Data Loading**: Imported large datasets into Spark DataFrames for distributed processing.
2. **Data Cleaning**:
   - Dropped columns with >50% missing values.
   - Replaced missing values (median for numerical, "Unknown" for categorical).
   - Capped outliers at 1st and 99th percentiles.
3. **Feature Engineering**:
   - Vectorized features for machine learning.
4. **Visualization**:
   - Scatter plots, heatmaps, distribution plots, and more for insights.

## Modeling

### K-Means Clustering
- **Objective**: Group borrowers based on financial behavior.
- **Implementation**: Used PySpark MLlib's KMeans algorithm.

### ElasticNet Regression
- **Objective**: Predict loan default probability.
- **Performance Metric**: Area Under Curve (AUC) = 0.78.

## Results

1. **Clustering**: Segmented borrowers to identify high-risk groups.
2. **Predictive Insights**:
   - Income-to-debt ratios and demographic patterns are significant predictors.
   - Achieved robust model performance with AUC of 0.78.

## Future Scope

- **Model Enhancements**:
  - Incorporate ensemble methods like Random Forest or Gradient Boosted Trees.
  - Explore deep learning models for improved accuracy.
- **Real-Time Processing**:
  - Implement streaming data analysis with Apache Kafka.
- **Feature Enrichment**:
  - Add external data sources like social media behavior.
- **Scalability**:
  - Optimize for larger datasets and GPU acceleration.

## References

1. [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
2. [PySpark MLlib](https://spark.apache.org/docs/latest/api/python/)
3. [Home Credit Dataset](https://www.kaggle.com/c/home-credit-default-risk/data)
4. [ElasticNet Regression](https://scikit-learn.org/stable/modules/linear_model.html#elastic-net)

## Contributors

- **Manavkumar Patel**
- **Pallavi Savant**
- **Bhavikk Shah**
- **Maram Reddy Srinath Reddy**
- **Satwika Sriram**

---
For more details, check the [Project Repository](https://github.com/pallavisavant/CSP_554/tree/main).
