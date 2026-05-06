"""
Module 6 Week B — Integration: NER + Embeddings Semantic Pipeline

Build an end-to-end NLP pipeline that combines named entity recognition
(Week A) with embedding-based semantic search (Week B) on a climate
article corpus.
"""

import numpy as np
import pandas as pd
import spacy
import torch


def load_and_preprocess(filepath):
    """Load the climate articles dataset and prepare texts for processing.

    Args:
        filepath: Path to the CSV file (e.g., 'data/climate_articles.csv').

    Returns:
        pandas DataFrame with at least columns: 'text', plus any
        preprocessing columns you add (e.g., cleaned text).
    """
    # TODO: Load the CSV, handle missing values, ensure text column is clean
    df = pd.read_csv(filepath)

    # حذف القيم الفارغة
    df = df.dropna(subset=["text"])

    # تنظيف النصوص
    df["text"] = df["text"].str.strip()

    # فلترة اللغة (اختياري لكن مهم)
    if "language" in df.columns:
        df = df[df["language"] == "en"]

    df = df.reset_index(drop=True)

    return df


def run_ner(texts):
    """Run named entity recognition on a list of texts using spaCy.

    Args:
        texts: List of strings to process.

    Returns:
        pandas DataFrame with columns: 'text_index', 'entity_text',
        'entity_label'. Each row is one extracted entity.
    """
    # TODO: Load a spaCy model, process each text, extract entities,
    #       and collect into a DataFrame
    nlp = spacy.load("en_core_web_sm")

    data = []

    for i, text in enumerate(texts):
        doc = nlp(text)

        for ent in doc.ents:
            data.append({
                "text_index": i,
                "entity_text": ent.text,
                "entity_label": ent.label_
            })

    return pd.DataFrame(data)


def compute_embeddings(texts, tokenizer, model):
    """Compute DistilBERT embeddings for a list of texts.

    Tokenize each text, pass through the model, and mean-pool the
    last hidden state to produce a single vector per text.

    Args:
        texts: List of strings.
        tokenizer: Hugging Face tokenizer.
        model: Hugging Face model.

    Returns:
        numpy array of shape (n_texts, 768).
    """
    import torch
    # TODO: Iterate over texts, tokenize with padding/truncation,
    #       run model forward pass (with torch.no_grad()), mean-pool hidden states
    import torch

    embeddings = []

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden_state = outputs.last_hidden_state  # (1, seq_len, 768)

        # mean pooling
        emb = last_hidden_state.mean(dim=1).squeeze().numpy()

        embeddings.append(emb)

    return np.array(embeddings)


def semantic_search(query, corpus_embeddings, corpus_texts, top_k=5):
    """Find the top-k most similar texts to the query using cosine similarity.

    Args:
        query: numpy array of shape (768,) — the query embedding.
        corpus_embeddings: numpy array of shape (n, 768) — corpus embeddings.
        corpus_texts: List of strings — the original texts.
        top_k: Number of results to return.

    Returns:
        List of (text, similarity_score) tuples, sorted by similarity descending.
    """
    # TODO: Compute cosine similarity between query and all corpus embeddings,
    #       sort by similarity, return top-k results
    similarities = np.dot(corpus_embeddings, query) / (
        np.linalg.norm(corpus_embeddings, axis=1) * np.linalg.norm(query)
    )

    # ترتيب النتائج
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append((corpus_texts[idx], similarities[idx]))

    return results


def enrich_with_entities(search_results, entity_df, corpus_texts):
    """Enrich semantic search results with NER entities.

    For each search result, look up its position in corpus_texts to get the
    text_index, then attach the matching entities from entity_df.

    Args:
        search_results: List of (text, score) tuples from semantic_search.
        entity_df: DataFrame from run_ner with columns:
                   'text_index', 'entity_text', 'entity_label'.
        corpus_texts: List of strings — the original corpus passed to
                      run_ner. Used to map a result text to its text_index.

    Returns:
        List of dictionaries, each with keys:
        'text', 'similarity', 'entities' (list of {'text': ..., 'label': ...}).
    """
    # TODO: For each (text, score) in search_results, find the text's
    #       position in corpus_texts (this is the text_index).
    # TODO: Filter entity_df to rows where text_index matches, then build
    #       a list of {'text': entity_text, 'label': entity_label} dicts.
    # TODO: Return one dict per search result with keys text, similarity,
    #       entities.
    enriched_results = []

    for text, score in search_results:
        text_index = corpus_texts.index(text)

        ents = entity_df[entity_df["text_index"] == text_index]

        entities = [
            {"text": row["entity_text"], "label": row["entity_label"]}
            for _, row in ents.iterrows()
        ]

        enriched_results.append({
            "text": text,
            "similarity": score,
            "entities": entities
        })

    return enriched_results


def demonstrate_pipeline(corpus_df, entity_df, embeddings, queries,
                         tokenizer, model):
    """Run the full pipeline demonstration on example queries.

    For each query string:
    1. Compute the query embedding (using the injected tokenizer and model)
    2. Perform semantic search against the corpus embeddings
    3. Enrich results with entities

    Args:
        corpus_df: DataFrame from load_and_preprocess.
        entity_df: DataFrame from run_ner.
        embeddings: numpy array of shape (n, 768) from compute_embeddings.
        queries: List of query strings.
        tokenizer: Hugging Face tokenizer (already loaded by the caller).
        model: Hugging Face model in eval mode (already loaded by the caller).

    Returns:
        Dictionary mapping each query string to its enriched results list.
    """
    # TODO: For each query, compute the query embedding by calling
    #       compute_embeddings([query], tokenizer, model)[0].
    # TODO: Call semantic_search with the query embedding and the corpus.
    # TODO: Call enrich_with_entities, passing corpus_df['text'].tolist()
    #       as corpus_texts.
    # TODO: Collect into a dict keyed by the query string and return it.
    results = {}

    corpus_texts = corpus_df["text"].tolist()

    for query in queries:
        # embedding للـ query
        query_emb = compute_embeddings([query], tokenizer, model)[0]

        # semantic search
        search_results = semantic_search(
            query_emb,
            embeddings,
            corpus_texts
        )

        # enrichment
        enriched = enrich_with_entities(
            search_results,
            entity_df,
            corpus_texts
        )

        results[query] = enriched

    return results


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModel

    # Load and preprocess
    df = load_and_preprocess("data/climate_articles.csv")
    if df is not None:
        texts = df["text"].tolist()
        print(f"Loaded {len(texts)} texts")

        # NER
        entities = run_ner(texts)
        if entities is not None:
            print(f"Extracted {len(entities)} entities")

        # Embeddings
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModel.from_pretrained("distilbert-base-uncased")
        model.eval()
        embs = compute_embeddings(texts, tokenizer, model)
        if embs is not None:
            print(f"Embedding matrix shape: {embs.shape}")

        # Demo queries
        with open("data/example_queries.txt") as f:
            queries = [line.strip() for line in f if line.strip()]

        if embs is not None and entities is not None:
            results = demonstrate_pipeline(
                df, entities, embs, queries, tokenizer, model
            )
            if results:
                for q, enriched in results.items():
                    print(f"\nQuery: {q}")
                    for r in enriched[:3]:
                        print(f"  Score: {r['similarity']:.4f}")
                        print(f"  Text: {r['text'][:100]}...")
                        print(f"  Entities: {r['entities'][:5]}")
