# main.py

import csv
from rag import RAG

def load_chunks_from_csv(path):
    chunks = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chunks.append({
                "id": row["id"],
                "text": row["text"],
                "source": row["source"]
            })
    return chunks

if __name__ == "__main__":
    chunks = load_chunks_from_csv("data/05_corpus_rag.csv")
    rag = RAG(chunks=chunks)

    print("--- Test hors-corpus ---")
    print(rag.answer_question("Quelle est la capitale du Japon ?"))

    print("\n--- Test contradiction ---")
    print(rag.answer_question("Le chat de Bob est vert, non ?"))