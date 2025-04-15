import pandas as pd
import numpy as np
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from rapidfuzz import fuzz

# Configure logging (optional).
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def smart_string_match(
    requested_col: str,
    available_cols: List[str],
    threshold: int = 75
) -> Optional[str]:
    """
    Matches the requested column name to the closest column name in available_cols
    using fuzzy matching. Returns the top match if score >= threshold, else None.
    """
    if not requested_col or not available_cols:
        return None

    def normalize(s: str) -> str:
        return re.sub(r"[\s_]", "", s).lower()

    requested_norm = normalize(requested_col)
    scored_matches = []

    for col in available_cols:
        col_norm = normalize(col)
        # Compute multiple fuzzy scores
        ratio_score = fuzz.ratio(requested_norm, col_norm)
        partial_score = fuzz.partial_ratio(requested_norm, col_norm)
        token_score = fuzz.token_sort_ratio(requested_norm, col_norm)
        # Weighted score
        weighted = (0.4 * ratio_score + 0.3 * partial_score + 0.3 * token_score)
        scored_matches.append((col, weighted))

    # Sort by descending score
    scored_matches.sort(key=lambda x: x[1], reverse=True)

    # Return top match if above threshold
    if scored_matches and scored_matches[0][1] >= threshold:
        return scored_matches[0][0]

    return None

def auto_convert_float_like_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempts to convert string columns containing float-like strings
    (with commas or dots) to numeric. Columns that fail remain unchanged.
    """
    df = df.copy()
    for col in df.select_dtypes(include='object'):
        try:
            df[col] = df[col].str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            logging.debug(f"Column '{col}' could not be converted to float.")
    return df

def choose_numeric_impute_strategy(series: pd.Series) -> str:
    """
    Automatically decide between 'mean' or 'median'
    based on the skewness of the numeric column.

    If |skew| > 2 => 'median'
    Otherwise => 'mean'
    """
    valid_vals = series.dropna()
    if valid_vals.empty:
        return "mean"  # fallback if there are no values at all

    skew_val = valid_vals.skew()
    if abs(skew_val) > 2:
        return "median"
    return "mean"

def data_curation(
    file_path: str,
    llm_response: Dict[str, List[str]]
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    1. Loads data from a CSV or Excel file (no chunking).
    2. Converts float-like strings to numeric.
    3. Filters/keeps columns based on llm_response (numeric, categorical, target).
    4. Imputes missing values:
       - Numeric columns: auto selects mean or median based on skewness.
       - Categorical columns: uses mode.
    5. Caps outliers in numeric columns using the 1.5 * IQR rule.
    6. Returns (df, corrected_llm_response) so that you can also see
       what columns were actually matched for each category.

    Parameters
    ----------
    file_path : str
        Path to the CSV or Excel file.
    llm_response : Dict[str, List[str]]
        e.g. {
          "numeric": ["Carrier Amount", "Term Date"],
          "categorical": ["POD", "Operation"],
          "target": ["Profit"],
          "model": "XGBoost"
        }

    Returns
    -------
    (df, corrected_llm_response)
      df : pd.DataFrame
          Curated DataFrame with matched columns, imputed missing values, outliers capped.
      corrected_llm_response : Dict[str, List[str]]
          Same structure as llm_response, but with updated column names that exist in df.
    """
    # 1. Load data
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Only CSV and Excel are supported.")

    # 2. Convert float-like strings to numeric
    df = auto_convert_float_like_strings(df)
    original_cols = df.columns.tolist()

    if not llm_response:
        raise ValueError("llm_response is required for curation.")

    # --- Parse the llm_response for each type of column
    numeric_requested = llm_response.get("numeric", [])
    categorical_requested = llm_response.get("categorical", [])
    target_requested = llm_response.get("target", [])

    # --- Fuzzy-match each requested column to real columns
    numeric_matched = [smart_string_match(col, original_cols) for col in numeric_requested]
    categorical_matched = [smart_string_match(col, original_cols) for col in categorical_requested]
    target_matched = [smart_string_match(col, original_cols) for col in target_requested]

    # Filter out None matches
    numeric_matched = [c for c in numeric_matched if c is not None]
    categorical_matched = [c for c in categorical_matched if c is not None]
    target_matched = [c for c in target_matched if c is not None]

    # Combine them for final DataFrame columns
    matched_cols = numeric_matched + categorical_matched + target_matched
    df = df[matched_cols]

    # For further steps, we'll base numeric/categorical on matched lists.
    # 4. Fill missing values in numeric columns
    for col in numeric_matched:
        if col in df.columns:  # just in case
            n_missing = df[col].isnull().sum()
            if n_missing > 0:
                strategy = choose_numeric_impute_strategy(df[col])
                fill_val = df[col].mean() if strategy == "mean" else df[col].median()
                df[col] = df[col].fillna(fill_val)
                logging.info(
                    f"Imputed {n_missing} missing values in numeric column '{col}' "
                    f"using {strategy}={round(fill_val, 3)}."
                )

    # 4b. Fill missing values in categorical columns
    for col in categorical_matched:
        if col in df.columns:
            n_missing = df[col].isnull().sum()
            if n_missing > 0:
                mode_vals = df[col].mode(dropna=True)
                fill_val = mode_vals[0] if not mode_vals.empty else "missing"
                df[col] = df[col].fillna(fill_val)
                logging.info(
                    f"Imputed {n_missing} missing values in categorical column '{col}' "
                    f"using mode='{fill_val}'."
                )

    # 5. Outlier capping for numeric columns (1.5 IQR rule)
    for col in numeric_matched:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            df[col] = np.clip(df[col], lower, upper)

    # Build a corrected llm_response with the matched columns
    corrected_llm_response = dict(llm_response)  # shallow copy
    corrected_llm_response["numeric"] = numeric_matched
    corrected_llm_response["categorical"] = categorical_matched
    corrected_llm_response["target"] = target_matched

    return df, corrected_llm_response

def data_ozet(
    file_path: str,
    head_rows: int = 5,
    max_unique: int = 20,
    sample_size: int = 5
) -> Dict[str, Any]:
    """
    Provides a summary of the data: shape, column summaries, head,
    potential categoricals, and cardinality info.
    """
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Only CSV and Excel are supported.")

    def detect_potential_categoricals(
        df_: pd.DataFrame,
        max_unique_: int = 20,
        rel_unique_thresh: float = 0.05
    ):
        potential = []
        cardinality_info = {}

        for c in df_.columns:
            data_ = df_[c]
            nuniq = data_.nunique(dropna=True)
            cardinality_info[c] = nuniq
            n_total = len(data_)

            if data_.dtype == "object":
                try:
                    converted = pd.to_numeric(data_.str.replace(",", ".", regex=False))
                    if converted.nunique() <= max_unique_ or (converted.nunique() / n_total) < rel_unique_thresh:
                        potential.append({
                            "column": c,
                            "reason": "object-but-floatlike-low-unique",
                            "n_unique": int(converted.nunique())
                        })
                    continue
                except ValueError:
                    potential.append({
                        "column": c,
                        "reason": "object-non-numeric",
                        "n_unique": int(nuniq)
                    })
                    continue

            if pd.api.types.is_numeric_dtype(data_):
                if nuniq <= max_unique_ or (nuniq / n_total) < rel_unique_thresh:
                    potential.append({
                        "column": c,
                        "reason": "numeric-low-unique",
                        "n_unique": int(nuniq)
                    })

        return potential, cardinality_info

    def column_summary(col_name: str) -> Dict[str, Any]:
        data_ = df[col_name]
        summary = {
            "dtype": str(data_.dtype),
            "missing_ratio": round(data_.isnull().mean(), 3),
            "n_unique": int(data_.nunique()),
        }
        if pd.api.types.is_numeric_dtype(data_):
            if not data_.isnull().all():
                summary.update({
                    "mean": round(data_.mean(), 3),
                    "std": round(data_.std(), 3),
                    "min": round(data_.min(), 3),
                    "max": round(data_.max(), 3),
                    "q1": round(data_.quantile(0.25), 3),
                    "median": round(data_.median(), 3),
                    "q3": round(data_.quantile(0.75), 3),
                })
            else:
                summary.update({
                    "mean": None, "std": None, "min": None,
                    "max": None, "q1": None, "median": None, "q3": None
                })
        else:
            sample_vals = data_.dropna().astype(str).unique()[:sample_size]
            summary["sample_values"] = list(sample_vals)
        return summary

    # Detect potential categoricals
    potential_cats, card_info = detect_potential_categoricals(df, max_unique)

    # Build column summaries
    col_summaries = {c: column_summary(c) for c in df.columns}

    return {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "column_summaries": col_summaries,
        "head": df.head(head_rows).to_dict(orient="records"),
        "potential_categoricals": potential_cats,
        "cardinality_info": card_info,
    }
