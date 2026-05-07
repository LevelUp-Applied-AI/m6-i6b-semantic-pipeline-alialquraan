import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


def load_and_preprocess(filepath):
    
    df = pd.read_csv(filepath)

    df = df.dropna(subset=["text"])

    df["text"] = df["text"].str.strip()

    en_df = df[df["language"] == "en"].reset_index(drop=True)
    ar_df = df[df["language"] == "ar"].reset_index(drop=True)

    english_texts = en_df["text"].head(10).tolist()
    arabic_texts = ar_df["text"].head(10).tolist()

    return english_texts, arabic_texts


def compute_embeddings(texts, tokenizer, model):
    
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

        last_hidden_state = outputs.last_hidden_state

        emb = last_hidden_state.mean(dim=1).squeeze().numpy()

        embeddings.append(emb)

    return np.array(embeddings)


def compute_similarity_matrix(embeddings):
    
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

    normalized_embeddings = embeddings / norms

    similarity_matrix = np.dot(
        normalized_embeddings,
        normalized_embeddings.T
    )

    return similarity_matrix


def create_heatmap(similarity_matrix, labels, output_path):
  
    plt.figure(figsize=(12, 10))

    plt.imshow(similarity_matrix)

    plt.colorbar()

    plt.xticks(
        range(len(labels)),
        labels,
        rotation=90,
        fontsize=6
    )

    plt.yticks(
        range(len(labels)),
        labels,
        fontsize=6
    )

    plt.title("Cross-Lingual Embedding Similarity Heatmap")

    plt.tight_layout()

    plt.savefig(output_path, dpi=300)

    plt.close()


def compare_cross_lingual_similarity(
    similarity_matrix,
    english_texts,
    arabic_texts
):
    
    print("\nCross-Lingual Similarity Examples:\n")

    for i in range(min(3, len(english_texts))):
        for j in range(min(3, len(arabic_texts))):

            similarity = similarity_matrix[i][len(english_texts) + j]

            print(f"English Text {i + 1} ↔ Arabic Text {j + 1}")
            print(f"Similarity: {similarity:.4f}")

            print(
                f"EN: {english_texts[i][:60]}..."
            )

            print(
                f"AR: {arabic_texts[j][:60]}..."
            )

            print("-" * 60)


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModel

    english_texts, arabic_texts = load_and_preprocess(
        "data/climate_articles.csv"
    )

    print(f"Loaded {len(english_texts)} English texts")
    print(f"Loaded {len(arabic_texts)} Arabic texts")

    all_texts = english_texts + arabic_texts

    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-multilingual-cased"
    )

    model = AutoModel.from_pretrained(
        "bert-base-multilingual-cased"
    )

    model.eval()

    embeddings = compute_embeddings(
        all_texts,
        tokenizer,
        model
    )

    print(f"Embedding matrix shape: {embeddings.shape}")

    similarity_matrix = compute_similarity_matrix(
        embeddings
    )

    print(
        f"Similarity matrix shape: {similarity_matrix.shape}"
    )

    labels = []

    for text in all_texts:
        labels.append(
            text[:40].replace("\n", " ")
        )

    create_heatmap(
        similarity_matrix,
        labels,
        "heatmap.png"
    )

    print("Saved heatmap.png")

    compare_cross_lingual_similarity(
        similarity_matrix,
        english_texts,
        arabic_texts
    )