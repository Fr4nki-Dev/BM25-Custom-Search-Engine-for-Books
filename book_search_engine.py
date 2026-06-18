import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import os
import sys
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# Set NLTK data path for bundled data
nltk.data.path.append(os.path.join(resource_path('nltk_data')))
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('maxent_ne_chunker_tab', quiet=True)
nltk.download('words', quiet=True)

#print(nltk.data.path)


class BookSearchEngine:
    def __init__(self, csv_file, k1=1.5, b=0.75, pivot_factor=1.0, field_weights=None):
        self.df = pd.read_csv(csv_file)
        self.k1 = k1
        self.b = b
        self.pivot_factor = pivot_factor
        self.field_weights = field_weights or {
            'Book Title': 10.0,
            'Book Author': 10.0,
            'Plot Summary': 0.5,
            'Book genres': 5.0
        }
        # Normalize weights to sum to 1
        total_weight = sum(self.field_weights.values())
        self.field_weights = {k: v / total_weight for k, v in self.field_weights.items()}
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.inverted_index = {}
        self.bm25 = None
        self.lsi_model = None
        self.tfidf_vectorizer = None
        self.doc_vectors = None
        self.preprocess_documents()
        self.build_index()
        self.apply_lsi()

    def preprocess_text(self, text):
        if pd.isna(text):
            return []
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens
                  if token not in self.stop_words and token.isalnum()]
        return tokens

    def preprocess_documents(self):
        self.documents = []
        self.doc_lengths = []
        self.weighted_doc_lengths = []

        for _, row in self.df.iterrows():
            doc_tokens = []
            weighted_length = 0
            for field, weight in self.field_weights.items():
                field_value = str(row.get(field, ''))
                if field == 'Book genres':
                    genres = ' '.join(self.parse_genres(field_value))
                    tokens = self.preprocess_text(genres)
                else:
                    tokens = self.preprocess_text(field_value)
                doc_tokens.extend(tokens)
                weighted_length += len(tokens) * weight
            self.documents.append(doc_tokens)
            self.doc_lengths.append(len(doc_tokens))
            self.weighted_doc_lengths.append(weighted_length)

        self.avg_doc_length = np.mean(self.doc_lengths) if self.doc_lengths else 1
        self.avg_weighted_doc_length = np.mean(self.weighted_doc_lengths) if self.weighted_doc_lengths else 1

    def build_index(self):
        self.bm25 = BM25Okapi(self.documents, k1=self.k1, b=self.b)
        adjusted_lengths = []
        for length in self.weighted_doc_lengths:
            adjusted = length / (self.pivot_factor + (1 - self.pivot_factor) * (length / self.avg_weighted_doc_length))
            adjusted_lengths.append(adjusted)
        self.doc_lengths = adjusted_lengths

    def apply_lsi(self, n_components=100):
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            token_pattern=None,
            lowercase=False
        )
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.documents)
        self.lsi_model = TruncatedSVD(n_components=n_components)
        self.doc_vectors = self.lsi_model.fit_transform(tfidf_matrix)

    def query_processing(self, query):
        # Tokenize and POS tag the query
        tokens = word_tokenize(query.lower())
        pos_tags = nltk.pos_tag(tokens)

        # Perform NER using NLTK
        chunked = nltk.ne_chunk(pos_tags)
        entity_filtered_list = []

        # Map NLTK entity labels to match original expected labels
        nltk_to_spacy_labels = {
            'PERSON': 'PERSON',
            'ORGANIZATION': 'ORG',
            'GPE': 'GPE'
        }

        for subtree in chunked:
            if hasattr(subtree, 'label') and subtree.label() in nltk_to_spacy_labels:
                entity_text = ' '.join(word for word, tag in subtree.leaves()).lower()
                entity_filtered_list.append(entity_text)

        # Existing query expansion logic
        expanded_tokens = []
        for token in tokens:
            expanded_tokens.append(token)
            synsets = wordnet.synsets(token)
            for syn in synsets:
                for lemma in syn.lemmas():
                    expanded_tokens.append(lemma.name().lower())

        combined_token_list = list(set(expanded_tokens + entity_filtered_list))
        return combined_token_list

    def query(self, query_text, top_k=10):
        query_text_clean = query_text.strip().lower()
        query_tokens = self.query_processing(query_text)
        if not query_tokens:
            return []

        # Initialize scores
        bm25_scores = self.bm25.get_scores(query_tokens)
        query_vector = self.tfidf_vectorizer.transform([query_tokens])
        query_lsi = self.lsi_model.transform(query_vector)
        lsi_scores = np.dot(self.doc_vectors, query_lsi.T).flatten()
        combined_scores = 0.7 * bm25_scores + 0.3 * lsi_scores

        # Check for exact and partial matches on title and author
        exact_match_boost = np.zeros(len(self.df))
        partial_match_boost = np.zeros(len(self.df))
        EXACT_MATCH_BOOST_VALUE = 10
        PARTIAL_MATCH_BOOST_VALUE = 5

        query_words = set(word_tokenize(query_text_clean))

        for idx, row in self.df.iterrows():
            title = str(row['Book Title']).lower() if not pd.isna(row['Book Title']) else ''
            author = str(row['Book Author']).lower() if not pd.isna(row['Book Author']) else ''

            if query_text_clean == title or query_text_clean == author:
                exact_match_boost[idx] = EXACT_MATCH_BOOST_VALUE
                continue

            title_words = set(word_tokenize(title)) if title else set()
            author_words = set(word_tokenize(author)) if author else set()
            title_overlap = len(query_words.intersection(title_words))
            author_overlap = len(query_words.intersection(author_words))

            if (query_text_clean in title or query_text_clean in author or
                    title_overlap > 0 or author_overlap > 0):
                partial_boost = PARTIAL_MATCH_BOOST_VALUE * max(title_overlap, author_overlap)
                partial_match_boost[idx] = partial_boost

        final_scores = combined_scores + exact_match_boost + partial_match_boost

        top_indices = np.argsort(final_scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            results.append({
                'Wikipedia ID': row['Wikipedia ID'],
                'Title': row['Book Title'],
                'Author': row['Book Author'],
                'Genres': self.parse_genres(row['Book genres']),
                'Summary': row['Plot Summary'],
                'Score': final_scores[idx]
            })
        return results

    def parse_genres(self, genres_str):
        if pd.isna(genres_str):
            return []
        try:
            genres = genres_str.strip('{}').split(',')
            genre_names = []
            for g in genres:
                parts = g.strip().split(':')
                if len(parts) > 1:
                    name = parts[-1].strip('"\n ')
                    if name:
                        genre_names.append(name)
            return genre_names
        except:
            return []


class BookSearchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Book Search Engine")
        self.root.geometry("800x600")
        self.engine = None
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Helvetica", 14))
        self.style.configure("TLabel", font=("Helvetica", 14))
        self.style.configure("TCombobox", font=("Helvetica", 14))

        # Query frame
        self.query_frame = ttk.Frame(self.root)
        self.query_frame.pack(pady=10, padx=10, fill=tk.X)

        # Query label and entry
        self.query_label = ttk.Label(self.query_frame, text="Enter your query:")
        self.query_label.pack(side=tk.LEFT)

        self.query_entry = ttk.Entry(self.query_frame, width=40)
        self.query_entry.pack(side=tk.LEFT, padx=5)
        self.query_entry.bind("<Return>", self.search)

        # Dropdown for number of results
        self.top_k_label = ttk.Label(self.query_frame, text="Results:")
        self.top_k_label.pack(side=tk.LEFT, padx=5)

        self.top_k_var = tk.StringVar(value="5")  # Default to 5
        self.top_k_dropdown = ttk.Combobox(
            self.query_frame,
            textvariable=self.top_k_var,
            values=[str(i) for i in range(1, 11)] + ["15", "20"],  # Options: 1 to 10, 15, 20
            width=5,
            state="readonly"
        )
        self.top_k_dropdown.pack(side=tk.LEFT, padx=5)

        # Search button
        self.search_button = ttk.Button(self.query_frame, text="Search", command=self.search)
        self.search_button.pack(side=tk.LEFT)
        self.search_button.config(state='disabled')

        # Status label
        self.status_label = ttk.Label(self.root, text="")
        self.status_label.pack(pady=5)

        # Content frame
        self.content_frame = ttk.Frame(self.root)
        self.content_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # Loading label
        self.loading_label = ttk.Label(
            self.content_frame,
            text="Loading...",
            font=("Helvetica", 28, "bold"),
            anchor="center"
        )
        self.loading_label.place(relx=0.5, rely=0.4, anchor="center")

        # Results text area
        self.results_text = scrolledtext.ScrolledText(
            self.content_frame, wrap=tk.WORD, width=90, height=25, font=("Helvetica", 11)
        )
        self.results_text.config(state='disabled')

        # Start engine initialization in background
        threading.Thread(target=self.initialize_engine, daemon=True).start()

    def initialize_engine(self):
        try:
            self.engine = BookSearchEngine(
                csv_file=resource_path('booksummaries.csv'),
                k1=0.75,
                b=0.9,
                pivot_factor=1.0,
                field_weights={
                    'Book Title': 10.0,
                    'Book Author': 10.0,
                    'Plot Summary': 0.5,
                    'Book genres': 5.0
                }
            )
            self.root.after(0, self.engine_ready)
        except Exception as e:
            self.root.after(0, self.display_error, f"Initialization failed: {str(e)}")

    def engine_ready(self):
        self.loading_label.destroy()
        self.results_text.pack(fill=tk.BOTH, expand=True)
        self.results_text.config(state='normal')
        self.results_text.insert(tk.END, "Ready to search.\n")
        self.results_text.config(state='disabled')
        self.search_button.config(state='normal')

    def search(self, event=None):
        if not self.engine:
            self.status_label.config(text="Engine not ready. Please wait.")
            return
        query = self.query_entry.get().strip()
        if not query:
            self.status_label.config(text="Please enter a query.")
            return
        try:
            top_k = int(self.top_k_var.get())
            if top_k <= 0:
                raise ValueError
        except ValueError:
            self.status_label.config(text="Please select a valid number of results.")
            return
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state='disabled')
        self.status_label.config(text="Searching...")
        self.search_button.config(state='disabled')
        self.root.update()
        threading.Thread(target=self.perform_search, args=(query, top_k), daemon=True).start()

    def perform_search(self, query, top_k):
        try:
            results = self.engine.query(query, top_k=top_k)
            self.root.after(0, self.display_results, results)
        except Exception as e:
            self.root.after(0, self.display_error, str(e))
        finally:
            self.root.after(0, self.reset_status)

    def display_results(self, results):
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        if not results:
            self.results_text.insert(tk.END, "No results found.\n")
        else:
            for i, res in enumerate(results, 1):
                self.results_text.insert(tk.END, f"Result {i}:\n")
                self.results_text.insert(tk.END, f"Title: {res['Title']}\n")
                self.results_text.insert(tk.END, f"Author: {res['Author']}\n")
                self.results_text.insert(tk.END, f"Genres: {', '.join(res['Genres'])}\n")
                self.results_text.insert(tk.END, f"Summary: {res['Summary']}\n")
                self.results_text.insert(tk.END, f"Score: {res['Score']:.4f}\n")
                self.results_text.insert(tk.END, "-" * 80 + "\n\n")
        self.results_text.config(state='disabled')

    def display_error(self, error_msg):
        self.loading_label.destroy()
        if not self.results_text.winfo_ismapped():
            self.results_text.pack(fill=tk.BOTH, expand=True)
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Error: {error_msg}\n")
        self.results_text.config(state='disabled')

    def reset_status(self):
        self.status_label.config(text="")
        self.search_button.config(state='normal')


def main():
    root = tk.Tk()
    app = BookSearchGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()