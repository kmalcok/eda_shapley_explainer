from llm import col_and_model_select, aciklama_uret
from entity_util import data_ozet, data_curation
from train import data_preparation, train_selected_model
from shp import get_shap_explainer
import shap
import pandas as pd

def run_pipeline(file_path, user_prompt):
    # 1. Veri Özeti
    print("📊 Veriseti özeti oluşturuluyor...")
    summary = data_ozet(file_path)

    # 2. LLM ile kolon ve model seçimi
    print("🧠 LLM ile kolon ve model seçimi yapılıyor...")
    selection_spec = col_and_model_select(summary, user_prompt)

    # 3. Veriyi hazırla (clean + match)
    print("🧼 Veri kürasyonu yapılıyor...")
    curated_df, fixed_spec = data_curation(file_path, selection_spec)

    # 4. Veriyi encode et ve eğitime hazır hale getir
    print("📦 Eğitim verisi hazırlanıyor...")
    prep_result = data_preparation(curated_df, fixed_spec)

    # 5. GridSearchCV ile model eğitimi
    print("🏋️ Model eğitiliyor...")
    train_result = train_selected_model(prep_result)

    model = train_result.get("model")
    X_train = train_result.get("X_train")
    model_type = train_result.get("model_type")

    # 6. SHAP analizi
    print("🔍 SHAP analizi yapılıyor...")
    explainer = get_shap_explainer(model, X_train, model_type)
    shap_values = explainer(X_train)

    # 7. SHAP değerlerini normalize et
    if shap_values.values.ndim == 3:
        shap_array = abs(shap_values.values).mean(axis=2)
    else:
        shap_array = abs(shap_values.values)

    shap_df = pd.DataFrame(shap_array, columns=train_result['features'])
    mean_abs_shap = shap_df.mean()
    top_features = mean_abs_shap.sort_values(ascending=False).head(3).index.tolist()

    # 8. LLM ile doğal dil açıklama üret
    print("📝 Açıklama oluşturuluyor...")
    explanation = aciklama_uret(mean_abs_shap.to_dict(), top_features, fixed_spec['target'][0], user_prompt)

    print("Yorum", explanation)
    # 9. Sonuçları döndür
    return {
        "selected_columns": fixed_spec,
        "train_scores": {
            "train_score": train_result['train_score'],
            "test_score": train_result['test_score']
        },
        "top_features": top_features,
        "explanation": explanation,
        "shap_importance": mean_abs_shap.to_dict()
    }

