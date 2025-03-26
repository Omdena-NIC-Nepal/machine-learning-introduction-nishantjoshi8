import pandas as pd
import numpy as np
from src.data_preprocessing import preprocess_data

def test_preprocess_data():
    # Test case 1: Check if the function handles missing values correctly
    df = pd.DataFrame({
        'CRIM': [0.1, 0.2, np.nan, 0.4],
        'ZN': [0, 0, 0, 0],
        'INDUS': [6.0, 7.0, 8.0, np.nan],
        'CHAS': [0, 1, 0, 1],
        'NOX': [0.5, 0.6, 0.7, 0.8],
        'RM': [6.0, 6.5, np.nan, 7.0],
        'AGE': [65.2, 64.0, 58.0, 80.0],
        'DIS': [4.0, 3.5, 2.5, 2.0],
        'RAD': [1, 2, 3, 4],
        'TAX': [300, 400, 500, 600],
        'PTRATIO': [15.3, 17.0, 18.0, 19.0],
        'B': [396.9, 396.0, 393.0, 395.0],
        'LSTAT': [4.0, 5.0, 6.0, 7.0],
        'MEDV': [24.0, 21.6, 34.7, 33.4]
    })
    
    processed_df = preprocess_data(df)
    
    # Check if missing values are filled
    assert processed_df.isnull().sum().sum() == 0, "Missing values were not handled correctly."
    
    # Test case 2: Check if categorical variables are encoded
    assert 'CHAS' in processed_df.columns, "Categorical variable 'CHAS' was not encoded correctly."
    
    # Test case 3: Check normalization of numerical features
    numerical_cols = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    for col in numerical_cols:
        assert processed_df[col].min() >= 0 and processed_df[col].max() <= 1, f"Feature '{col}' was not normalized correctly."
    
    print("All tests passed!")