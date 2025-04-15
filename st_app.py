import streamlit as st
import pandas as pd
import tempfile
from pipeline_test import run_pipeline  # run_pipeline fonksiyonunun olduğu dosya

st.set_page_config(page_title="SHAP Data Analyzer", layout="wide")

st.title("🔍 SHAP Destekli Veri Analizi")

# Layout
left, right = st.columns([2, 3])

# 📁 Dosya yükleme
uploaded_file = right.file_uploader("📂 CSV Dosyasını Yükle veya Sürükle Bırak", type=["csv"])

# 🧠 Prompt input
user_prompt = right.text_area("🧠 Prompt girin", placeholder="Örn: Bu veride kaliteyi etkileyen faktörleri analiz et...", height=100)

# Veri önizlemesi (solda)
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    left.subheader("📊 Veri Önizlemesi")
    left.dataframe(df.head(20), use_container_width=True)

    # Temp dosyaya yaz → pipeline string yol istiyor
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    # Analiz butonu
    if right.button("🚀 Analizi Başlat"):
        with st.spinner("Model eğitiliyor ve SHAP analizi yapılıyor..."):
            try:
                result = run_pipeline(tmp_path, user_prompt)

                # Sonuç gösterimi
                st.success("✅ Analiz tamamlandı!")
                st.markdown("### 📝 Açıklama:")
                st.write(result["explanation"])

                st.markdown("### 📌 En Önemli Özellikler:")
                st.write(result["top_features"])

                st.markdown("### 📈 SHAP Özellik Önem Skorları:")
                st.dataframe(pd.DataFrame.from_dict(result["shap_importance"], orient="index", columns=["SHAP Score"]).sort_values("SHAP Score", ascending=False))

                st.markdown("### 🎯 Model Skorları:")
                st.json(result["train_scores"])

            except Exception as e:
                st.error(f"Hata oluştu: {e}")
else:
    left.info("🧾 Bir CSV dosyası yüklediğinizde burada veri önizlemesini göreceksiniz.")
