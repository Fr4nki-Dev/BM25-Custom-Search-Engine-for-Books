# BM25 Book Search Engine

A Python desktop application for searching book summaries using a hybrid information retrieval pipeline. The system combines BM25 ranking, weighted metadata fields, query expansion, and Latent Semantic Indexing (LSI) to return relevant books from a structured book-summary dataset.

## Project Overview

This project implements a book retrieval engine for searching by title, author, genre, and plot description. It is designed for queries such as:

```text
fantasy books with dragons
books about a heist
George Orwell
romance novels about family secrets
```

The application provides a simple Tkinter interface where users can enter a query and choose how many ranked results to return.

## Key Features

- BM25-based document ranking
- Field weighting for title, author, genre, and plot summary
- Pivoted document length normalisation logic
- TF-IDF and Latent Semantic Indexing using Truncated SVD
- Query expansion using WordNet synonyms
- Named entity extraction using NLTK
- Exact and partial boosts for title and author matches
- Tkinter desktop interface
- Evaluation notebook for Precision@k, Recall@k, and F1-score@k

## Files

```text
README.md
requirements.txt
.gitignore
book_search_engine.py
booksummaries.csv
retrieval_evaluation.ipynb
project_overview.md
```

## Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- WordNet
- rank-bm25
- Scikit-learn
- Tkinter
- Matplotlib
- Jupyter Notebook

## How the Search Engine Works

1. **Load the dataset**  
   The engine reads book metadata and plot summaries from `booksummaries.csv`.

2. **Pre-process text**  
   Text is lowercased, punctuation is removed, stop words are filtered out, and tokens are lemmatised.

3. **Build document representations**  
   The system combines weighted fields from book title, author, plot summary, and genre.

4. **Rank documents**  
   BM25 provides the main relevance score, while LSI adds a semantic similarity component.

5. **Boost title and author matches**  
   Exact and partial matches on title or author receive additional score boosts.

6. **Display ranked results**  
   The GUI shows the title, author, genres, summary, and relevance score for each result.

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application from the repository root:

```bash
python book_search_engine.py
```

The first run may download required NLTK resources automatically.

## Evaluation

The evaluation notebook is included as:

```text
retrieval_evaluation.ipynb
```

It demonstrates how to calculate:

- Precision@k
- Recall@k
- F1-score@k
- Average scores by query type


