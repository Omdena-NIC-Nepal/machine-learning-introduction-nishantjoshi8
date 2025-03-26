import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)

def handle_missing_values(df):
    """Handle missing values in the dataset."""
    # Example: Fill missing values with the median for numerical features
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        df[column].fillna(df[column].median(), inplace=True)
    # Example: Fill missing values for categorical features with the mode
    for column in df.select_dtypes(include=['object']).columns:
        df[column].fillna(df[column].mode()[0], inplace=True)
    return df

def encode_features(df, categorical_features):
    """Encode categorical features using OneHotEncoder."""
    encoder = OneHotEncoder(sparse=False, drop='first')
    encoded_features = encoder.fit_transform(df[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
    df = df.drop(categorical_features, axis=1)
    return pd.concat([df, encoded_df], axis=1)

def normalize_features(df, numerical_features):
    """Normalize numerical features using StandardScaler."""
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    return df

def split_data(df, target_column, test_size=0.2, random_state=42):
    """Split the dataset into training and testing sets."""
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def preprocess_data(file_path, target_column, categorical_features, numerical_features):
    """Complete preprocessing pipeline."""
    df = load_data(file_path)
    df = handle_missing_values(df)
    df = encode_features(df, categorical_features)
    df = normalize_features(df, numerical_features)
    return split_data(df, target_column)