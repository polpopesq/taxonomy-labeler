import torch
from sentence_transformers import SentenceTransformer, util
from file_reader import read_data

model = SentenceTransformer('paraphrase-mpnet-base-v2')
taxonomy, data, taxonomy_dict = read_data()

labels = list(taxonomy_dict.keys()) if taxonomy_dict else taxonomy
descriptions = list(taxonomy_dict.values()) if taxonomy_dict else taxonomy
emb_descriptions = model.encode(descriptions, convert_to_tensor=True)

def classify_rows_vectorized(df, top_n=3, batch_size=64, progress_callback=None):
    texts = df['info'].fillna('').tolist()
    all_embs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb_batch = model.encode(batch, convert_to_tensor=True)
        all_embs.append(emb_batch)
        if progress_callback:
            progress_callback(min(i + batch_size, len(texts)), len(texts))

    all_embs = torch.cat(all_embs, dim=0)
    cos_scores = util.cos_sim(all_embs, emb_descriptions)
    top_scores, top_idxs = torch.topk(cos_scores, k=top_n, dim=1)

    for i in range(top_n):
        df[f'top{i+1}_label'] = [labels[idx] for idx in top_idxs[:, i].cpu().numpy()]
        df[f'top{i+1}_score'] = (top_scores[:, i].cpu().numpy() * 100).round(2)

    return df.drop(columns=['info'], errors='ignore')

def classify_preview(top_n=3, preview_rows=50):
    preview = data.head(preview_rows).copy()
    return classify_rows_vectorized(preview, top_n=top_n)

def classify_full(top_n=3, progress_callback=None):
    full = data.head(150).copy()
    classified = classify_rows_vectorized(full, top_n=top_n, progress_callback=progress_callback)

    def combine_labels(row):
        labels = [
            row[f'top{i+1}_label']
            for i in range(top_n)
            if row[f'top{i+1}_score'] >= 25
        ]
        return ", ".join(labels)

    classified['labels'] = classified.apply(combine_labels, axis=1)

    columns_to_keep = ['description', 'business_tags', 'sector', 'category', 'niche', 'labels']
    return classified[columns_to_keep]

