import os
import json, re
import openai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")
client = openai.OpenAI(
    api_key=api_key,
)

def col_and_model_select(data_ozet, prompt):
    with open('example.json', 'r') as f:
        exm = json.load(f)
    with open('template.json', 'r') as f:
        tmp = json.load(f)

    def request_spec(ozet, user_prompt):
        return client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f"""
                    ##Main Task : 
                    - You are responsible for, selecting columns from given dataset and deciding whether use a mlp model or xgboost model.
                    - Selected columns will be used in a ml model and after that correlation will be analysed. 
                    - User prompt will be like \"why ı am making bad profits\", \"why this companies capital growth is negative some months\" etc.
                    - Select the target column via the user prompt. 
                    - You dont have to select all columns.
                    - ## If data looks bad, consider it will be curated after you choosing columns. 
                    - ## Only choose columns that we will need.
                    - ## Give exact names from dataset.
                    You should follow this format strictly:
                    {tmp}
                    An example usage : 
                    {exm}
                    ##Table Metadata:
                    {ozet}
                    #User Prompt:
                    {user_prompt}
                    You only give json formatted columns names no markdown no explanations no another text.
                    """
                },
                {
                    "role": "system",
                    "content": "You are an expert data science assistant. Responsible at making column decisions for ml model that will be used in shap correlation analysis. You only give json formatted columns names no markdown no explanations no another text."
                }
            ]
        ).choices[0].message.content

    def fix_malformed_json(broken_text):
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "The following text is supposed to be JSON but is malformed. Fix it and only return valid JSON. Do not explain anything."
                },
                {
                    "role": "user",
                    "content": broken_text
                }
            ]
        )
        return response.choices[0].message.content

    # İlk istek
    raw_spec = request_spec(data_ozet, prompt)
    raw_spec_clean = re.sub(r"```json|```", "", raw_spec).strip()

    try:
        return json.loads(raw_spec_clean)
    except json.JSONDecodeError:
        print("⚠️ LLM çıktısı parse edilemedi, otomatik düzeltiliyor...")

        fixed_raw = fix_malformed_json(raw_spec_clean)
        fixed_clean = re.sub(r"```json|```", "", fixed_raw).strip()

        try:
            return json.loads(fixed_clean)
        except json.JSONDecodeError:
            print("❌ Düzeltme de başarısız oldu. Kodu elle kontrol etmen gerek.")
            raise

def aciklama_uret(shap_importance_dict, top_features, target, prompt):
    import pandas as pd

    # Dict → DataFrame
    df = pd.DataFrame.from_dict(shap_importance_dict, orient='index', columns=['SHAP Importance'])
    df = df.loc[top_features].sort_values(by='SHAP Importance', ascending=False)

    full_prompt = f"""
You will be given SHAP-based feature importance values.
User prompt (You will answer to this): {prompt}
Target column: {target}
Top 3 most important features based on SHAP values: {', '.join(top_features)}

Feature importance (SHAP mean values):
{df.to_string()}

Write a plain and clear explanation for the user based on the feature contributions.
Your response should help non-technical users understand which features matter most and why.
"""
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.4
    )
    return response.choices[0].message.content.strip()
