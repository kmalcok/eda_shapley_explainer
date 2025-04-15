from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

class SmartCategoricalEncoder:
    def __init__(self, model_type: str, cat_cols: list):
        self.model_type = model_type.lower()
        self.cat_cols = cat_cols
        self.encoder = None

    def fit(self, X: pd.DataFrame, y=None):
        if not self.cat_cols:
            self.encoder = None
            return self

        if self.model_type == "xgboost":
            self.encoder = ColumnTransformer(
                transformers=[
                    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), self.cat_cols)
                ],
                remainder="passthrough"
            )
        elif self.model_type == "mlp":
            self.encoder = ColumnTransformer(
                transformers=[
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.cat_cols)
                ],
                remainder="passthrough"
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.encoder.fit(X)
        return self

    def transform(self, X: pd.DataFrame):
        if self.encoder is None:
            X.columns = X.columns.str.replace(r"[\[\]<>]", "_", regex=True)
            return X
        X.columns = X.columns.str.replace(r"[\[\]<>]", "_", regex=True)
        return pd.DataFrame(self.encoder.transform(X),
                            columns=self.get_feature_names(),
                            index=X.index)

    def fit_transform(self, X: pd.DataFrame, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names(self):
        if not self.encoder:
            return self.cat_cols
        try:
            return self.encoder.get_feature_names_out()
        except:
            return [f"f_{i}" for i in range(self.encoder.transform(np.zeros((1, len(self.cat_cols)))).shape[1])]
