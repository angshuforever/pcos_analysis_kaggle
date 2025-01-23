# PCOS Prediction Project Guide

## Project Setup

1. Create a new project directory
```bash
mkdir pcos_prediction
cd pcos_prediction
```

2. Create a virtual environment and activate it
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages
```bash
pip install pandas numpy scikit-learn matplotlib
```

## Project Structure

Create the following files:
```
pcos_prediction/
├── data/
│   └── pcos_prediction_dataset.csv
├── src/
│   └── pcos_analysis.py
└── README.md
```

## Implementation Steps

1. **Data Loading and Exploration**
   - Import required libraries
   - Load dataset using pandas
   - Implement `explore_data()` function to analyze:
     - Dataset shape
     - Column information
     - Missing values
     - Basic statistics

2. **Data Preprocessing**
   - Handle missing values (`handle_missing_values()`)
     - Remove columns with > 30% missing values
     - Impute remaining missing values
   - Handle outliers (`handle_bad_data()`)
     - Use IQR method
     - Replace outliers with boundary values

3. **Feature Engineering**
   - Implement `select_features()`
     - Handle non-numeric columns
     - Scale features using StandardScaler
     - Select best features using f_regression
   - Create final dataset using selected features

4. **Model Development**
   - Split data into training and testing sets
   - Train Linear Regression model
   - Evaluate model performance using R² score
   - Visualize feature importance

## Running the Project

1. Place your dataset in the data directory:
```bash
mv pcos_prediction_dataset.csv data/
```

2. Run the analysis:
```bash
python src/pcos_analysis.py
```

## Key Functions

### explore_data(df)
Performs initial data exploration and prints summary statistics.

### handle_missing_values(df, threshold=30)
- Input: DataFrame and threshold for maximum missing values
- Output: Cleaned DataFrame with handled missing values

### handle_bad_data(df)
- Input: DataFrame
- Output: DataFrame with outliers handled using IQR method

### select_features(X, y, n_features=10)
- Input: Feature matrix X, target variable y, number of features to select
- Output: Selected features, transformed data, and feature scores

### train_and_evaluate_model(X_train, X_test, y_train, y_test, selected_features)
- Input: Training and test data, feature names
- Output: Trained model, train score, test score

## Expected Output

The script will:
1. Print data exploration results
2. Show missing value statistics
3. Display selected features and their importance scores
4. Print model performance metrics
5. Generate feature importance visualization

## Notes

- Ensure target variable is encoded as numeric before feature selection
- Adjust `threshold` parameter in `handle_missing_values()` based on your data quality
- Monitor warnings about non-numeric columns during feature selection
