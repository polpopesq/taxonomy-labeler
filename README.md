## taxonomy-labeler
# Problem 
Accept a list of companies(csv document) with the associated data:
description, business_tags, sector ,category ,niche
Receive a static taxonomy (a list of labels) relevant to the insurance industry.

# Label examples:
Risk Assessment Services
Crisis Management Services
Community Engagement Services
Stakeholder Services
Corporate Responsibility Services

Build a solution that accurately classifies these companies, and any similar ones, into one or more labels from the static taxonomy.

# Thought process
First of all, a very important factor is that the initial data is not labeled, so we cannot train classic models in the first phase of classification. This reduces the accuracy of the algorithm because we do not have a clear measurement of success(ground truth).
In evaluating approaches, I compared two main strategies:
Zero-shot classification using facebook/bart-large-mnli and Semantic similarity scoring using Sentence-BERT (SBERT) and cosine similarity

# Initial Findings
Through empirical testing on a representative dataset, I found that the SBERT + cosine similarity method was approximately 700 times faster than the zero-shot classification pipeline. The BART-based zero-shot model is significantly slower due to its autoregressive nature and the need to process each text-label pair independently, which makes it unsuitable for real-time or large-scale batch processing.
As a result, I initially opted for the SBERT-based approach due to its superior inference speed and scalability.
While SBERT provided an efficient framework, early results indicated low cosine similarity scores and suboptimal label matches, largely due to the use of short, context-free labels in the taxonomy and the limited semantic capacity of smaller SBERT variants like all-MiniLM-L6-v2

# Optimization
To make the model work better, I also cleaned and standardized the input data, in the data reader file. I combined all important text fields (like description, business_tags, sector, etc.) into a single column(‘info’) and removed common marketing buzzwords such as "cutting-edge" or "best-in-class." These types of phrases don’t add real meaning and can confuse the model. This step helps ensure that the input focuses on useful, specific details about each company, making the similarity comparison more accurate and consistent.
Combining all the useful, standardized data into the ‘info’ column gives more context than using the description alone, which was often too short or vague. The model has more information to work with, helping it understand what each company does, which improved the results.
Of course this is not the most accurate solution. This method is faster and works well given our limits — we don’t have labeled data to train a model properly, and everything runs on a local machine (not on the cloud or a GPU server). So I chose this setup mainly for speed and simplicity, not perfect accuracy.
That said, this isn’t the most accurate way to classify companies. Because the model isn’t trained on this specific data, and the labels are pretty general, the results can still be a bit off sometimes. But for now, it’s a good starting point and extensible framework for classifying insurance-related companies in the absence of labeled training data.

# Demo
To demonstrate how the algorithm works I created a simple Streamlit app to visualize results better. The app contains a preview of the first 50 companies with top 3 labels and a button to process and export the whole labeled dataset.

# Installation
Make sure you have Python 3.8+ installed
1. Clone the repository:
   git clone https://github.com/your-username/taxonomy-labeler.git
   cd taxonomy-labeler
2. Create a virtual environment (optional but recommended):
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
3. Install the required dependencies:
   pip install -r requirements.txt

# Run the App
Start the Streamlit application with:
  streamlit run app.py

# Output
After processing, you can download a CSV with the following structure:
description, business_tags, sector, category, niche, labels
"Insurance platform for wildfire risk", "property, risk", "Insurtech", ..., "Risk Assessment Services, Crisis Management Services"

#Git Cleanup Recommendation
Add the following to .gitignore:
  __pycache__/
  *.pyc
  .idea/
  .venv/
