# main.py

import csv
from rag import RAG


def load_chunks_from_csv(path):
    """Charge uniquement les textes des chunks (liste de str),
    format attendu par la Brique 1 (VectorDB._creer)."""
    chunks = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chunks.append(row["text"])
    return chunks


if __name__ == "__main__":
    chunks = load_chunks_from_csv("data/05_corpus_rag.csv")
    rag = RAG(chunks=chunks)

    print("--- Test hors-corpus ---")
    print(rag.answer_question("Quelle est la capitale du Japon ?"))

    print("\n--- Test contradiction ---")
    print(rag.answer_question("Le chat de Bob est vert, non ?"))