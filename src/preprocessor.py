from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# currency conversion
class CurrencyConverter(BaseEstimator, TransformerMixin):
    def __init__(self, rates=None, base_currency="INR"):
        if rates is None:
            rates = {"INR": 1.0, "USD": 83.0, "EUR": 90.0}
        self.rates = rates
        self.base_currency = base_currency

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["amount_converted"] = X.apply(
            lambda row: row["amount"] * self.rates.get(row["currency"], 1.0), axis=1
        )
        return X
    
    def get_feature_names_out(self, input_features=None):
        """Add 'amount_converted' to the feature names"""
        if input_features is None:
            input_features = []
        # Return original features + new feature
        output_features = list(input_features) + ['amount_converted']
        return np.asarray(output_features, dtype=object)


# typo fixing   
class TypoFixer(BaseEstimator, TransformerMixin):
    def __init__(self, column='merchant_category', typo_map=None):
        if typo_map is None:
            typo_map = {'Groceires': 'Groceries'}
        self.column = column
        self.typo_map = typo_map
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if self.column in X.columns:
            X[self.column] = X[self.column].replace(self.typo_map)
        return X
    
    def get_feature_names_out(self, input_features=None):
        """Feature names unchanged - just fixing typos"""
        if input_features is None:
            return np.array([self.column], dtype=object)
        return np.asarray(input_features, dtype=object)
    

# outlier clipping
class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, features=None, lower_quantile=0.01, upper_quantile=0.99):
        self.features = features
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.bounds = {}

    def fit(self, X, y=None):
        for col in self.features:
            q_low = X[col].quantile(self.lower_quantile)
            q_high = X[col].quantile(self.upper_quantile)
            self.bounds[col] = (q_low, q_high)
        return self

    def transform(self, X):
        X = X.copy()
        for col, (low, high) in self.bounds.items():
            X[col] = X[col].clip(lower=low, upper=high)
        return X
    
    def get_feature_names_out(self, input_features=None):
        """Feature names unchanged - just clipping values"""
        if input_features is None:
            return np.asarray(self.features, dtype=object)
        return np.asarray(input_features, dtype=object)
    

# Define Feature Groups
num_features = [
    "amount_converted", "velocity", "ip_risk_score", "customer_age",
    "account_tenure", "geo_distance", "merchant_risk_score", "failed_login_attempts"
]

cat_features = [
    "currency", "merchant_category", "transaction_type", "channel", "location"
]

bin_features = ["card_present", "is_international"]


# Pipelines for Each Feature Type

# Numerical pipeline
num_pipeline = Pipeline([
    ("outlier_clipper", OutlierClipper(features=num_features)),
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical pipeline
cat_pipeline = Pipeline([
    ("typo_fixer", TypoFixer()),
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

cat_pipeline_catboost = Pipeline([
    ("typo fixer", TypoFixer()),
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown"))
])

# Binary pipeline
bin_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent"))
])


# Full Preprocessing Pipeline
def Preprocessor():
    preprocessor = Pipeline([
        ("currency_converter", CurrencyConverter()),
        ("transformer", ColumnTransformer([
            ("num", num_pipeline, num_features),
            ("cat", cat_pipeline, cat_features),
            ("bin", bin_pipeline, bin_features)
        ]))
    ])
    return preprocessor

def Preprocessor_catboost():
    preprocessor_catboost = Pipeline([
        ("currency_converter", CurrencyConverter()),
        ("transformer", ColumnTransformer([
            ("num", num_pipeline, num_features),
            ("cat", cat_pipeline_catboost, cat_features),
            ("bin", bin_pipeline, bin_features)
        ]))
    ])
    return preprocessor_catboost