# Elevvo_pathways_internship
Welcome to my Elevvo Internship repository! This repo documents the core data science tasks I completed as part of my internship, using Jupyter Notebook.

# INTERNSHIP TASK OVERVIEW

✅Task 1 – Sentiment Analysis

Explored sentiment analysis using 50,000 IMDb movie reviews.

Preprocessed text, converted reviews into numerical features using TF-IDF, and trained a Logistic Regression model achieving ~88% accuracy.

Visualized the most common words in positive vs negative reviews to gain insights into emotional expression in writing.


✅Task 2 – News Category Classification

Classified news articles into World, Sports, Business, and Science/Technology using the AG News dataset.

Used Naive Bayes with TF-IDF features, achieving ~89% accuracy.

Created word clouds to visualize terms most associated with each news category.


✅Task 3 – Fake News Detection

Combined real and fake news datasets from Kaggle.

Cleaned text, transformed into numerical vectors, and trained classifiers like Logistic Regression and SVM, achieving >90% accuracy.

Evaluated with F1-score, confusion matrices, and word clouds to reveal differences in vocabulary between fake and real news.


✅Task 4 – Named Entity Recognition (NER)

Identified entities such as PERSON, ORG, LOC in text.

Tried both rule-based detection and model-based NER using spaCy’s pretrained models (en_core_web_sm and en_core_web_trf).

Visualized entities with Displacy for easy interpretation.


✅Task 5 – Question Answering with Transformers

Built a QA system using Hugging Face’s DistilBERT.

Tested on Nigerian Pidgin and Yoruba passages from Wikipedia, and my own short story Arà the Owl.

Compared DistilBERT vs TinyRoBERTa, evaluating with Exact Match (66.7%) and F1-score (88.9%) on Yoruba text.





# TECH STACKS & TOOLS:

Programming & Environment: Python, Google Colab
Data Handling: Pandas, NumPy
Feature Extraction: TF-IDF
ML Models: Logistic Regression, Random Forest, Naive Bayes, XGBoost

NLP Libraries:

scikit-learn: preprocessing, vectorization, model training

spaCy: Named Entity Recognition (NER)

Hugging Face Transformers: Question Answering (DistilBERT, TinyRoBERTa)


Evaluation Tools: Accuracy, F1-score, Confusion Matrix, evaluate library (EM & F1 for QA)
Others: matplotlib (visualization)




# KEY LEARNINGS:

Building end-to-end NLP pipelines: preprocessing → feature extraction → model training

Understanding the differences between traditional ML and deep learning (Transformers)

Practical skills in text classification tasks: sentiment, news, fake news

NER: rule-based vs model-based methods, trade-offs, and visualization

Model evaluation: accuracy, F1-score, EM, confusion matrices, and speed vs accuracy trade-offs

Handling long contexts in Question Answering

Applying theory to real-world datasets and improving model robustness
