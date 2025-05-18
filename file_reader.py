import pandas as pd
import re
import os

REDUNDANT_PHRASES = [
    r"\bhigh quality\b",
    r"\bwe offer\b",
    r"\bour company (offers|provides)\b",
    r"\bstate[-\s]?of[-\s]?the[-\s]?art\b",
    r"\bcutting[-\s]?edge\b",
    r"\bbest[-\s]?in[-\s]?class\b",
    r"\bworld[-\s]?class\b",
    r"\bintegrated solutions\b",
    r"\bcomprehensive\b",
    r"\bmission[-\s]?critical\b",
    r"\bturnkey\b",
    r"\bvalue[-\s]?added\b",
    r"\bleading\b",
    r"\btrusted\b",
    r"\binnovative\b",
    r"\bstrategic\b",
    r"\bglobal\b",
    r"\bscalable\b"
]

def remove_redundant_phrases(text, patterns=REDUNDANT_PHRASES):
    text = str(text).lower()
    for phrase in patterns:
        text = re.sub(phrase, '', text)
    return re.sub(r'\s+', ' ', text).strip()

def standardize_data(df):
    text_columns = ['description', 'business_tags', 'sector', 'category', 'niche']
    for col in text_columns:
        if col not in df.columns:
            df[col] = ''
    df['info'] = df[text_columns].fillna('').agg(' '.join, axis=1)
    df['info'] = df['info'].apply(remove_redundant_phrases)
    return df

def read_data():
    temp_taxonomy = pd.read_excel("dataIN/insurance_taxonomy.xlsx")
    taxonomy = temp_taxonomy['label'].dropna().tolist()

    data = pd.read_csv("dataIN/ml_insurance_challenge.csv")

    # Ensure text fields are strings to avoid float/subscriptable errors
    text_columns = ['description', 'business_tags', 'sector', 'category', 'niche']
    for col in text_columns:
        if col in data.columns:
            data[col] = data[col].fillna("").astype(str)

    data = standardize_data(data)

    extended_path = "dataIN/Expanded_Taxonomy_Descriptions.csv"
    taxonomy_dict = None
    if os.path.exists(extended_path):
        df = pd.read_csv(extended_path)
        taxonomy_dict = dict(zip(df['label'], df['description']))

    return taxonomy, data, taxonomy_dict
