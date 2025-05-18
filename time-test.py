#fisier facut cu gpt sa vad cat de rpd se incarca fiecare, irelevant in app.py

import time
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import pandas as pd

taxonomy = ["Finanțe", "Tehnologie", "Sănătate", "Educație", "Transport"]
data = pd.DataFrame({
    "description": [
        "Această companie dezvoltă software pentru analiza datelor financiare.",
        "O firmă care oferă servicii medicale digitale pacienților din mediul rural.",
        "Startup educațional ce oferă cursuri interactive pentru programare.",
        "Companie de logistică ce optimizează transportul urban prin AI.",
        "Platformă de gestionare a investițiilor automatizate."
    ]
})

# 1. Zero-shot classification cu BART
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

start_bart = time.time()
for descriere in data['description']:
    _ = classifier(descriere, candidate_labels=taxonomy, multi_label=True)
end_bart = time.time()

# 2. SentenceTransformer + cos_sim
model = SentenceTransformer('all-MiniLM-L6-v2')
emb_etichete = model.encode(taxonomy, convert_to_tensor=True)

start_st = time.time()
for descriere in data['description']:
    emb_descriere = model.encode(descriere, convert_to_tensor=True)
    _ = util.cos_sim(emb_descriere, emb_etichete)[0]
end_st = time.time()

print(f"Timp BART (zero-shot): {end_bart - start_bart:.2f} sec")
print(f"Timp SentenceTransformer: {end_st - start_st:.2f} sec")
