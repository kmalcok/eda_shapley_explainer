from xgboost import XGBClassifier, XGBRegressor
import numpy as np
import time
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    RepeatedKFold,
    StratifiedKFold,
    KFold
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier, MLPRegressor

def model_selector(
        model_type,
        X_train,
        y_train,
        problem_type="auto",
        random_state=42,
        n_repeats=3
):

    if problem_type not in ["auto", "classification", "regression"]:
        raise ValueError("problem_type must be 'auto', 'classification', or 'regression'.")

    if problem_type == "auto":
        if np.issubdtype(y_train.dtype, np.number):
            unique_vals = np.unique(y_train)
            if len(unique_vals) > 0.2 * len(y_train):
                problem_type = "regression"
            else:
                problem_type = "classification"
        else:
            problem_type = "classification"

    if problem_type == "classification":
        if not np.issubdtype(y_train.dtype, np.number):
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)

    if problem_type == "classification":
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        num_classes = len(unique_classes)
        min_class_size = np.min(class_counts) if len(class_counts) > 0 else 1
    else:
        num_classes = None
        min_class_size = len(y_train)

    n_splits = max(2, min(5, min_class_size))

    if model_type not in ["mlp", "xgboost"]:
        raise ValueError("model_type must be either 'mlp' or 'xgboost'.")

    if model_type == "mlp":
        if problem_type == "classification":
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', MLPClassifier(max_iter=2500, random_state=random_state, verbose=True))
            ])
            param_grid = {
                'mlp__hidden_layer_sizes': [(64,), (128,), (64, 32)],
                'mlp__activation': ['relu', 'tanh'],
                'mlp__solver': ['adam'],
                'mlp__alpha': [0.0001, 0.001],
                'mlp__learning_rate': ['constant', 'adaptive'],
                'mlp__early_stopping': [True],
                'mlp__validation_fraction': [0.1, 0.2],
                'mlp__n_iter_no_change': [10, 20]
            }
        else:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', MLPRegressor(max_iter=2500, random_state=random_state, verbose=True))
            ])
            param_grid = {
                'mlp__hidden_layer_sizes': [(64,), (128,), (64, 32)],
                'mlp__activation': ['relu', 'tanh'],
                'mlp__solver': ['adam'],
                'mlp__alpha': [0.0001, 0.001],
                'mlp__learning_rate': ['constant', 'adaptive'],
                'mlp__early_stopping': [True],
                'mlp__validation_fraction': [0.1, 0.2],
                'mlp__n_iter_no_change': [15, 40]
            }
    else:
        if problem_type == "classification":
            pipeline = XGBClassifier(
                eval_metric='mlogloss',
                tree_method='hist',
                random_state=random_state,
                verbosity=2  # ⬅️ Eğitim süreci logları
            )
        else:
            pipeline = XGBRegressor(
                eval_metric='rmse',
                tree_method='hist',
                random_state=random_state,
                verbosity=2
            )

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'gamma': [0, 0.1],
            'min_child_weight': [1, 3]
        }

    if problem_type == "classification":
        try:
            cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, shuffle=True, random_state=random_state)
            _ = list(cv.split(X_train, y_train))
        except Exception:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        try:
            cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        except Exception:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scoring = {
        'accuracy': 'accuracy',
        'f1_macro': 'f1_macro'
    } if problem_type == "classification" else {
        'r2': 'r2',
        'neg_rmse': 'neg_root_mean_squared_error'
    }

    refit_metric = 'accuracy' if problem_type == "classification" else 'r2'

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        refit=refit_metric,
        cv=cv,
        n_jobs=-1,
        verbose=2,  # ⬅️ GridSearch eğitim çıktıları
        error_score=np.nan
    )

    start_time = time.time()
    grid.fit(X_train, y_train)
    duration = time.time() - start_time

    print("✅ En iyi parametreler:", grid.best_params_)
    print("✅ En iyi skor:", grid.best_score_)
    print("✅ Eğitim skoru:", grid.score(X_train, y_train))
    print("✅ En iyi model:", grid.best_estimator_)

    result = {
        "model_type": model_type,
        "problem_type": problem_type,
        "best_estimator": grid.best_estimator_,
        "best_params": grid.best_params_,
        "best_score": grid.best_score_,
        "cv_results": grid.cv_results_,
        "duration": duration,
        "folds_used": n_splits
    }

    if problem_type == "classification":
        result["num_classes"] = num_classes
    else:
        result["target_distribution"] = "continuous"

    return result
