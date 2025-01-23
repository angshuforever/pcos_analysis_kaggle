# Import required libraries

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def explore_data(df):
    print("Dataset Shape:", df.shape)
    print("\nDataset Info:")
    print(df.info())

    missing_values = df.isnull().sum()
    missing_percentages = (missing_values / len(df)) * 100
    print("\nMissing Values Summary:")
    print(missing_percentages[missing_percentages > 0])

    print("\nBasic Statistics:")
    print(df.describe())


def handle_missing_values(df, threshold=30):
    missing_percentages = (df.isnull().sum() / len(df)) * 100
    columns_to_drop = missing_percentages[missing_percentages > threshold].index
    df_cleaned = df.drop(columns=columns_to_drop)

    numeric_columns = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = df_cleaned.select_dtypes(include=['object']).columns

    # Numeric imputation
    numeric_imputer = SimpleImputer(strategy='median')
    df_cleaned[numeric_columns] = numeric_imputer.fit_transform(df_cleaned[numeric_columns])

    # Categorical imputation
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df_cleaned[categorical_columns] = categorical_imputer.fit_transform(df_cleaned[categorical_columns])

    return df_cleaned


def handle_bad_data(df):
    df_cleaned = df.copy()
    numeric_columns = df_cleaned.select_dtypes(include=['int64', 'float64']).columns

    for column in numeric_columns:
        Q1 = df_cleaned[column].quantile(0.25)
        Q3 = df_cleaned[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_cleaned.loc[df_cleaned[column] < lower_bound, column] = lower_bound
        df_cleaned.loc[df_cleaned[column] > upper_bound, column] = upper_bound

    return df_cleaned


def select_features(X, y, n_features=10):
    X_numeric = X.select_dtypes(include=['int64', 'float64'])

    non_numeric_columns = set(X.columns) - set(X_numeric.columns)
    if non_numeric_columns:
        print(f"Warning: The following non-numeric columns were removed: {non_numeric_columns}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)

    # Ensure y is numeric
    if not np.issubdtype(y.dtype, np.number):
        raise ValueError("Target variable `y` must be numeric. Please encode categorical values before proceeding.")

    selector = SelectKBest(score_func=f_regression, k=min(n_features, X_numeric.shape[1]))
    X_selected = selector.fit_transform(X_scaled, y)

    selected_features = X_numeric.columns[selector.get_support()].tolist()

    scores = pd.DataFrame({
        'Feature': X_numeric.columns,
        'Score': selector.scores_
    }).sort_values('Score', ascending=False)

    return selected_features, X_selected, scores


def prepare_final_dataset(df, selected_features, target_column):
    final_columns = selected_features + [target_column]
    final_df = df[final_columns].copy()

    return final_df


def train_and_evaluate_model(X_train, X_test, y_train, y_test, selected_features):
    model = LinearRegression()
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"Train R² Score: {train_score:.4f}")
    print(f"Test R² Score: {test_score:.4f}")

    plt.figure(figsize=(10, 6))
    feature_importance = pd.DataFrame({
        'Feature': selected_features,
        'Coefficient': abs(model.coef_)
    }).sort_values('Coefficient', ascending=True)

    plt.barh(feature_importance['Feature'], feature_importance['Coefficient'])
    plt.title('Feature Importance in Linear Regression Model')
    plt.xlabel('Absolute Coefficient Value')
    plt.tight_layout()
    plt.show()

    return model, train_score, test_score


def main():
    print("Step 1: Loading data...")
    df = pd.read_csv('../data/pcos_prediction_dataset.csv')

    print("\nStep 2: Exploring data...")
    explore_data(df)

    print("\nStep 3: Handling missing values...")
    df_cleaned = handle_missing_values(df)

    print("\nStep 4: Handling bad data...")
    df_cleaned = handle_bad_data(df_cleaned)

    print("\nStep 5: Preparing features and target...")
    target_column = 'Diagnosis'
    X = df_cleaned.drop(target_column, axis=1)
    y = df_cleaned[target_column]

    print("\nEncoding target variable `Diagnosis`...")
    if not np.issubdtype(y.dtype, np.number):
        le = LabelEncoder()
        y = le.fit_transform(y)  # Ensure `y` is numerical

    print("\nStep 6: Selecting features...")
    selected_features, X_selected, feature_scores = select_features(X, y)
    print("\nTop Selected Features:")
    print(feature_scores.head(10))

    print("\nStep 7: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )

    print("\nStep 8: Training and evaluating model...")
    model, train_score, test_score = train_and_evaluate_model(
        X_train, X_test, y_train, y_test, selected_features
    )

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()