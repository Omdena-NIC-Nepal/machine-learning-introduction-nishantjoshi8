import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

def create_interaction_features(df, feature1, feature2):
    """Create interaction features between two features."""
    df[f'{feature1}_x_{feature2}'] = df[feature1] * df[feature2]
    return df

def create_polynomial_features(df, feature, degree=2):
    """Create polynomial features for a given feature."""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(df[[feature]])
    poly_feature_names = poly.get_feature_names_out([feature])
    
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
    return pd.concat([df, poly_df], axis=1)

def drop_unnecessary_features(df, features_to_drop):
    """Drop unnecessary features from the DataFrame."""
    return df.drop(columns=features_to_drop)

def feature_engineering_pipeline(df):
    """Run the feature engineering pipeline."""
    # Example of creating interaction features
    df = create_interaction_features(df, 'RM', 'LSTAT')
    
    # Example of creating polynomial features
    df = create_polynomial_features(df, 'RM', degree=2)
    
    # Drop unnecessary features
    features_to_drop = ['Unnamed: 0']  # Example feature to drop
    df = drop_unnecessary_features(df, features_to_drop)
    
    return df