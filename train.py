from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
from encoder import SmartCategoricalEncoder
import grid_search

def data_preparation(df, selection_spec):
    target = selection_spec['target'][0]
    num_cols = selection_spec.get('numeric', [])
    cat_cols = selection_spec.get('categorical', [])
    model_type = selection_spec['model'].lower()

    # Kolonlar kontrolü
    eksik_kolonlar = [col for col in num_cols + cat_cols + [target] if col not in df.columns]
    if eksik_kolonlar:
        raise KeyError(f"Veride eksik kolon(lar) bulundu: {eksik_kolonlar}")

    # Gerekli kolonları seç
    df = df[num_cols + cat_cols + [target]].copy()
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    df[cat_cols] = df[cat_cols].fillna("missing")

    X = df[num_cols + cat_cols]
    y = df[target]

    # Target encoding
    if y.dtype == 'object' or y.nunique() < 20:
        y = LabelEncoder().fit_transform(y)

    # Kategorik encoder
    encoder = SmartCategoricalEncoder(model_type, cat_cols)
    X = encoder.fit_transform(X)

    # Sayısalları ayrıca scale et (sadece MLP için)
    scaler = None
    if model_type == "mlp":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Train/test böl
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "features": num_cols + cat_cols,
        "encoder": encoder,
        "scaler": scaler,
        "model_type": model_type
    }

def train_selected_model(prep_result):
    X_train = prep_result["X_train"]
    X_test = prep_result["X_test"]
    y_train = prep_result["y_train"]
    y_test = prep_result["y_test"]
    model_type = prep_result["model_type"]
    features = prep_result["features"]

    grid_result = grid_search.model_selector(model_type, X_train, y_train)

    # 🔧 'best_estimator' veya 'best_estimator_' varsa al
    model = grid_result.get("best_estimator") or grid_result.get("best_estimator_")

    if model is None:
        raise RuntimeError("GridSearch başarısız oldu, 'best_estimator' bulunamadı.")

    model.fit(X_train, y_train)

    return {
        "model": model,
        "X_train": X_train,
        "y_train": y_train,
        "train_score": model.score(X_train, y_train),
        "test_score": model.score(X_test, y_test),
        "features": features,
        "model_type": model_type
    }

