import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# pip install pandas numpy scikit-learn



def load_data(file_path, file_type="csv"):
    """
    Load any dataset into a Dask DataFrame.
    """
    if file_type == "csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    return df

def profile_data(df):
    """
    Profile the dataset to display schema, missing values, and basic statistics.
    """
    print("\nSchema:")
    print(df.dtypes)

    print("\nBasic Statistics:")
    print(df.describe())

    print("\nMissing Values:")
    print(df.isnull().mean())

def clean_data(df):
    """
    Clean the dataset by handling missing values and capping outliers.
    """
    # Replace missing numeric values with median and cap outliers
    for col in df.select_dtypes(include=[np.number]).columns:
        median = df[col].median()
        df[col] = df[col].fillna(median)

        # Cap outliers at the 99th percentile
        upper_limit = np.percentile(df[col].dropna(), 99)  # Use dropna to handle missing values
        df[col] = np.clip(df[col], None, upper_limit)

    # Replace missing categorical values with "UNKNOWN"
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("UNKNOWN")

    return df


def feature_engineering(df):
    """
    Add derived features dynamically based on numeric columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        df["FEATURE_SUM"] = df[numeric_cols].sum(axis=1)
        df["FEATURE_MEAN"] = df[numeric_cols].mean(axis=1)
    return df

def clustering(df, feature_columns, k=3):
    """
    Apply K-Means clustering using sklearn.
    """
    kmeans = KMeans(n_clusters=k, random_state=42)
    df["Cluster"] = kmeans.fit_predict(df[feature_columns])
    return df

def regression_model(df, feature_columns, label_column):
    """
    Train a linear regression model using sklearn.
    """
    X = df[feature_columns]
    y = df[label_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    print("\nRegression Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)

    # Model Evaluation
    score = model.score(X_test, y_test)
    print("R^2 Score:", score)

    return model



def main():
    file_path = "/Users/pallavi/PycharmProjects/bigdata/Salary_Data.csv"  # Update with the path to your dataset

    # Step 1: Load Data
    df = load_data(file_path)

    # Step 2: Profile Data
    profile_data(df)

    # Step 3: Clean Data
    df = clean_data(df)

    # Step 4: Feature Engineering
    df = feature_engineering(df)

    # Step 5: Clustering
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_columns = list(numeric_cols)
    df = clustering(df, feature_columns)

    # Step 6: Regression Model
    label_column = "Salary"  # Update with your target column
    feature_columns = [col for col in numeric_cols if col != label_column]
    regression_model(df, feature_columns, label_column)


if __name__ == "__main__":
    main()