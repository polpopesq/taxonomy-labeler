import streamlit as st
from io import BytesIO
from sbert_cos import classify_preview, classify_full

st.set_page_config(page_title="Insurance Taxonomy Classifier", layout="wide")
st.title("Insurance Taxonomy Classifier")

top_n = 3
preview_rows = 50

st.subheader(f"Previewing First {preview_rows} Companies (Top {top_n} Labels)")
preview_data = classify_preview(top_n=top_n, preview_rows=preview_rows)
columns = ['description'] + sum([[f'top{i+1}_label', f'top{i+1}_score'] for i in range(top_n)], [])
st.dataframe(preview_data[columns])

st.subheader("Download Full Labeled Dataset")
if st.button("Process and Prepare CSV"):
    progress_text = st.empty()
    progress_bar = st.progress(0.0)

    def update_progress(current, total):
        pct = current / total
        progress_bar.progress(pct)
        progress_text.markdown(f"Processing: {current} / {total} rows")

    full_data = classify_full(top_n=top_n, progress_callback=update_progress)
    csv_bytes = BytesIO()
    full_data.to_csv(csv_bytes, index=False)
    csv_bytes.seek(0)

    st.download_button(
        label="Download Full CSV",
        data=csv_bytes,
        file_name="labeled_companies_full.csv",
        mime="text/csv"
    )
