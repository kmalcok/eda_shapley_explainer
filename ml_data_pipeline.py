from entity_util import *

def data_curation(file_path):
    # 1. Veri yükleme
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Desteklenmeyen dosya formatı. Sadece CSV ve Excel desteklenir.")

    df = auto_convert_float_like_strings(df)
    original_df = df.copy()

    # 2. Sayısal ve kategorik ayır
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # 3. Eksik değer doldurma
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())
    for col in categorical_cols:
        if df[col].isnull().any():
            mode = df[col].mode()
            df[col] = df[col].fillna(mode[0] if not mode.empty else "missing")

    # 4. Uç değer düzeltme (winsorize)
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = (df[col] < lower) | (df[col] > upper)
        if outliers.any():
            df[col] = np.clip(df[col], lower, upper)

    return df


def auto_convert_float_like_strings(df):
    df = df.copy()
    for col in df.select_dtypes(include='object'):
        try:
            df[col] = df[col].str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col])
        except:
            continue
    return df

def validate_column(df, col):
    if col not in df.columns:
        return False
    if df[col].isnull().mean() > 0.5:
        return False
    if df[col].nunique() <= 1:
        return False
    if df[col].dtype == 'object':
        sample = df[col].dropna().astype(str).sample(min(5, len(df)))
        try:
            pd.to_numeric(sample.str.replace(",", "."), errors='raise')
        except:
            return False
    return True

#+
def fix_list(cols, df):
    available_cols = df.columns.tolist()
    matched = [smart_string_match(col, available_cols) for col in cols]
    return [col for col in matched if col and validate_column(df, col)]

